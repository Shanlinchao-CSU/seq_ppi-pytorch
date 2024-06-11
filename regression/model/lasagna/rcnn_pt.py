from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys

import scipy

if '../../../embeddings' not in sys.path:
    sys.path.append('../../../embeddings')

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.split(os.path.abspath(__file__))[0].rsplit('\\', 3)[0])
from seq2tensor import s2t

class Regression(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(Regression, self).__init__()
        self.hidden_dim = hidden_dim
        self.avg_pool_size = 5
        self.input_dim = dim
        self.first_pool_conv = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_dim, 3),
            nn.MaxPool1d(kernel_size=2)
        )
        self.pool_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim * 3, hidden_dim, 3),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv = nn.Conv1d(self.hidden_dim * 3, self.hidden_dim, 3)
        self.GRU = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.predict = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def shrink(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.first_pool_conv(x)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, self.GRU(x)[0]], dim=-1)
        x = torch.transpose(x, 1, 2)

        x = self.pool_conv(x)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, self.GRU(x)[0]], dim=-1)
        x = torch.transpose(x, 1, 2)

        x = self.pool_conv(x)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, self.GRU(x)[0]], dim=-1)
        x = torch.transpose(x, 1, 2)

        x = self.conv(x)
        x = nn.functional.avg_pool1d(x, x.shape[-1])
        x = torch.squeeze(x, dim=-1)
        return x

    def forward(self, x1, x2):
        x1 = self.shrink(x1)
        x2 = self.shrink(x2)
        merge_text = torch.mul(x1, x2)
        main_output = self.predict(merge_text)
        return main_output


class MyDataSet(Dataset):
    def __init__(self, x1, x2, label):
        self.x1 = x1
        self.x2 = x2
        self.label = label

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.label[idx]


# 蛋白id 对应的英文序列
id2seq_file = '../../../mtb/preprocessed/SKEMPI_seq.txt'
# key:蛋白id    value:index
id2index = {}
# 存储id2seq_file中的蛋白英文序列
seqs = []
index = 0
for line in open(id2seq_file):
    line = line.strip().split('\t')
    id2index[line[0]] = index
    seqs.append(line[1])
    index += 1
# 存储protein.actions.tsv中出现的蛋白的英文序列
seq_array = []
# 存储protein.dictionary.tsv中出现的所有蛋白id(key),value为新的index
id2_aid = {}
sid = 0

seq_size = 100
emb_files = ['../../../embeddings/default_onehot.txt', '../../../embeddings/string_vec5.txt',
             '../../../embeddings/CTCoding_onehot.txt', '../../../embeddings/vec5_CTC.txt']
ds_file = "../../../mtb/preprocessed/SKEMPI_all_dg_avg.txt"
label_index = -1
rst_file = "results/yeast_wvctc_rcnn_50_5.txt"
use_emb = 3
hidden_dim = 50
epochs = 1
sid1_index = 0
sid2_index = 1
use_log = 0

if len(sys.argv) > 1:
    ds_file, label_index, rst_file, use_emb, hidden_dim, epochs, use_log = sys.argv[1:]
    label_index = int(label_index)
    use_emb = int(use_emb)
    hidden_dim = int(hidden_dim)
    epochs = int(epochs)
    use_log = int(use_log)

seq2t = s2t(emb_files[use_emb])
max_data = -1
limit_data = max_data > 0
# 存储相互关系，蛋白质用id2_aid的value表示
raw_data = []
raw_ids = []
skip_head = True
x = None
count = 0
for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').replace('\t\t','\t').split('\t')
    raw_ids.append((line[sid1_index], line[sid2_index]))
    if id2_aid.get(line[sid1_index]) is None:
        id2_aid[line[sid1_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid1_index]]])
    line[sid1_index] = id2_aid[line[sid1_index]]
    if id2_aid.get(line[sid2_index]) is None:
        id2_aid[line[sid2_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid2_index]]])
    line[sid2_index] = id2_aid[line[sid2_index]]
    raw_data.append(line)
    if limit_data:
        count += 1
        if count >= max_data:
            break

len_m_seq = np.array([len(line.split()) for line in seq_array])
avg_m_seq = int(np.average(len_m_seq)) + 1
max_m_seq = max(len_m_seq)

dim = seq2t.dim
seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)])
seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

all_min, all_max = 99999999, -99999999
score_labels = np.zeros(len(raw_data))
for i in range(len(raw_data)):
    if use_log:
        score_labels[i] = np.log(float(raw_data[i][label_index]))
    else:
        score_labels[i] = float(raw_data[i][label_index])
    if score_labels[i] < all_min:
        all_min = score_labels[i]
    if score_labels[i] > all_max:
        all_max = score_labels[i]
for i in range(len(score_labels)):
    score_labels[i] = (score_labels[i] - all_min) / (all_max - all_min)

kf = KFold(n_splits=10, shuffle=True)
train_test = []
tries = 5
cur = 0
for train, test in kf.split(score_labels):
    train_test.append((train, test))
    cur += 1
    if cur >= tries:
        break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = nn.MSELoss().to(device)
batch_size = 256
total_train_step = 0
fold = 0

num_total = 0.
total_mse = 0.
total_mae = 0.
total_cov = 0.

def scale_back(v):
    if use_log:
        return np.exp(v * (all_max - all_min) + all_min)
    else:
        return v * (all_max - all_min) + all_min


# KFold
for train, test in train_test:
    merge_model = None
    merge_model = Regression(13, hidden_dim).to(device)
    optimizer = torch.optim.Adam(merge_model.parameters(), lr=0.001, eps=1e-6, amsgrad=True)
    fold += 1
    print("Fold {}:".format(fold))
    x1_data = seq_tensor[seq_index1[train]].astype(np.float32)
    x2_data = seq_tensor[seq_index2[train]].astype(np.float32)
    label_data = score_labels[train].astype(np.float32)
    train_dataset = MyDataSet(x1_data, x2_data, label_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    # Epoch
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        merge_model.train()
        for x1, x2, labels in train_dataloader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            output = merge_model(x1, x2).squeeze()
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
            # total_train_step += 1
            # if total_train_step % 20 == 0:
            #     print("训练次数：{}, loss: {}".format(total_train_step, loss.item()))
    merge_model.eval()
    with torch.no_grad():
        fp2 = open('records/pred_record.'+rst_file[rst_file.rfind('/')+1:], 'w')
        x1_test = torch.tensor(seq_tensor[seq_index1[test]], dtype=torch.float32).to(device)
        x2_test = torch.tensor(seq_tensor[seq_index2[test]], dtype=torch.float32).to(device)
        label_data = score_labels[test].astype(np.float32)
        output = merge_model(x1_test, x2_test).to('cpu')
        this_mae, this_mse, this_cov = 0., 0., 0.
        this_num_total = 0
        for i in range(len(label_data)):
            this_num_total += 1
            diff = abs(label_data[i] - output[i])
            this_mae += diff
            this_mse += diff**2
        num_total += this_num_total
        total_mae += this_mae
        total_mse += this_mse
        mse = total_mse / num_total
        mae = total_mae / num_total
        npf_output = output.numpy().squeeze()
        this_cov = scipy.stats.pearsonr(npf_output, label_data)[0]
        for i in range(len(test)):
            fp2.write(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index]) + '\t' + str(scale_back(npf_output[i])) + '\t' + str(scale_back(score_labels[test[i]])) + '\t' + str(npf_output[i]) + '\n')
            print(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index]) + '\t' + str(scale_back(npf_output[i])) + '\t' + str(scale_back(score_labels[test[i]])) + '\t' + str(npf_output[i]))
        total_cov += this_cov
        print (mse, mae, this_cov)
        fp2.close()

mse = total_mse / num_total
mae = total_mae / num_total
total_cov /= len(train_test)
print (mse, mae, total_cov)

with open(rst_file, 'w') as fp:
    fp.write('mae=' + str(mae) + '\nmse=' + str(mse) + '\ncorr=' + str(total_cov))