from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

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


class SP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(SP, self).__init__()
        self.hidden_dim = hidden_dim
        self.avg_pool_size = 5
        self.input_dim = dim
        self.first_pool_conv = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_dim, 3),
            nn.MaxPool1d(kernel_size=3)
        )
        self.pool_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim * 3, hidden_dim, 3),
            nn.MaxPool1d(kernel_size=3)
        )
        self.conv = nn.Conv1d(self.hidden_dim * 3, self.hidden_dim, 3)
        self.GRU = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.predict = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(100, int((self.hidden_dim + 7) / 2)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(int((self.hidden_dim + 7) / 2), 2)
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
        main_output = nn.functional.softmax(main_output, dim=-1)
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
id2seq_file = '../../../yeast/preprocessed/protein.dictionary.tsv'
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

seq_size = 2000
emb_files = ['../../../embeddings/default_onehot.txt', '../../../embeddings/string_vec5.txt',
             '../../../embeddings/CTCoding_onehot.txt', '../../../embeddings/vec5_CTC.txt']
ds_file = "../../../yeast/preprocessed/protein.actions.tsv"
label_index = -1
rst_file = "results/yeast_wvctc_rcnn_50_5.txt"
use_emb = 3
hidden_dim = 50
epochs = 1

if len(sys.argv) > 1:
    ds_file, label_index, rst_file, use_emb, hidden_dim, epochs = sys.argv[1:]
    label_index = int(label_index)
    use_emb = int(use_emb)
    hidden_dim = int(hidden_dim)
    epochs = int(epochs)

seq2t = s2t(emb_files[use_emb])
max_data = -1
limit_data = max_data > 0
# 存储相互关系，蛋白质用id2_aid的value表示
raw_data = []
skip_head = True
x = None
count = 0
for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').split('\t')
    if id2index.get(line[0]) is None or id2index.get(line[1]) is None:
        continue
    if id2_aid.get(line[0]) is None:
        id2_aid[line[0]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[0]]])
    line[0] = id2_aid[line[0]]
    if id2_aid.get(line[1]) is None:
        id2_aid[line[1]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[1]]])
    line[1] = id2_aid[line[1]]
    raw_data.append(line)
    if limit_data:
        count += 1
        if count >= max_data:
            break

len_m_seq = np.array([len(line.split()) for line in seq_array])
avg_m_seq = int(np.average(len_m_seq)) + 1
max_m_seq = max(len_m_seq)

dim = seq2t.dim
seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)], dtype='float32')
seq_index1 = np.array([line[0] for line in tqdm(raw_data)])
seq_index2 = np.array([line[1] for line in tqdm(raw_data)])

class_labels = np.zeros(len(raw_data))
for i in range(len(raw_data)):
    class_labels[i] = raw_data[i][2]

kf = KFold(n_splits=5, shuffle=True)
train_test = []
tries = 5
cur = 0
for train, test in kf.split(class_labels):
    train_test.append((train, test))
    cur += 1
    if cur >= tries:
        break

# Eval params
num_total = 0
num_hit = 0
num_pos = 0
num_true_pos = 0
num_false_pos = 0
num_true_neg = 0
num_false_neg = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = nn.CrossEntropyLoss()
loss_func.to(device)
batch_size = 256
total_train_step = 0
fold = 0
# KFold
for train, test in train_test:
    merge_model = None
    merge_model = SP(13, 50).to(device)
    optimizer = torch.optim.Adam(merge_model.parameters(), lr=0.0001, eps=1e-6, amsgrad=True)
    fold += 1
    print("Fold {}:".format(fold))
    x1_data = seq_tensor[seq_index1[train]]
    x2_data = seq_tensor[seq_index2[train]]
    label_data = class_labels[train].astype(np.int64)
    train_dataset = MyDataSet(x1_data, x2_data, label_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Epoch
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        merge_model.train()
        for x1, x2, labels in train_dataloader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            output = merge_model(x1, x2)
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
        fold_total = 0
        fold_hit = 0
        fold_pos = 0
        fold_true_pos = 0
        fold_false_pos = 0
        fold_true_neg = 0
        fold_false_neg = 0
        x1_test = torch.tensor(seq_tensor[seq_index1[test]]).to(device)
        x2_test = torch.tensor(seq_tensor[seq_index2[test]]).to(device)
        label_data = class_labels[test].astype(np.int32)
        output = merge_model(x1_test, x2_test).to('cpu')
        for i in range(len(output)):
            fold_total += 1
            if np.argmax(output[i]) == np.argmax(label_data[i]):
                fold_hit += 1
            if label_data[i] > 0:
                fold_pos += 1
                if output[i][0] < output[i][1]:
                    fold_true_pos += 1
                else:
                    fold_false_pos += 1
            else:
                if output[i][0] > output[i][1]:
                    fold_true_neg += 1
                else:
                    fold_false_neg += 1

        num_total += fold_total
        num_hit += fold_hit
        num_pos += fold_pos
        num_true_pos += fold_true_pos
        num_false_pos += fold_false_pos
        num_true_neg += fold_true_neg
        num_false_neg += fold_false_neg

        dm = ((fold_true_pos + fold_true_neg) * (fold_true_pos + fold_false_neg) * (fold_false_pos + fold_true_neg) * (
                fold_false_pos + fold_false_neg)) ** 0.5
        accuracy = fold_hit / fold_total if fold_total > 0 else 1
        prec = fold_true_pos / (fold_true_pos + fold_false_pos) if fold_true_pos + fold_false_pos > 0 else 1
        recall = fold_true_pos / fold_pos if fold_pos > 0 else 1
        spec = fold_true_neg / (fold_true_neg + fold_false_neg) if fold_true_neg + fold_false_neg > 0 else 1
        f1 = 2. * prec * recall / (prec + recall) if prec + recall > 0 else 1
        mcc = ((fold_true_pos * fold_true_neg - fold_false_pos * fold_false_neg) / dm) if dm > 0 else 1
        print("Accuracy:{}\nPrec:{}\nRecall:{}\nSpec:{}\nF1:{}\nMcc:{}\n".format(accuracy, prec, recall, spec, f1, mcc))

dm = ((num_true_pos + num_true_neg) * (num_true_pos + num_false_neg) * (num_false_pos + num_true_neg) * (
        num_false_pos + num_false_neg)) ** 0.5
accuracy = num_hit / num_total if num_total > 0 else 1
prec = num_true_pos / (num_true_pos + num_false_pos) if num_true_pos + num_false_pos > 0 else 1
recall = num_true_pos / num_pos if num_pos > 0 else 1
spec = num_true_neg / (num_true_neg + num_false_neg) if num_true_neg + num_false_neg > 0 else 1
f1 = 2. * prec * recall / (prec + recall) if prec + recall > 0 else 1
mcc = (num_true_pos * num_true_neg - num_false_pos * num_false_neg) / dm if dm > 0 else 1
print("Accuracy:{}\nPrec:{}\nRecall:{}\nSpec:{}\nF1:{}\nMcc:{}\n".format(accuracy, prec, recall, spec, f1, mcc))

with open(rst_file, 'w') as fp:
    fp.write('acc=' + str(accuracy) + '\tprec=' + str(prec) + '\trecall=' + str(recall) + '\tspec=' + str(
        spec) + '\tf1=' + str(f1) + '\tmcc=' + str(mcc))
