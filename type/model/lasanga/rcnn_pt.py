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
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.split(os.path.abspath(__file__))[0].rsplit('\\', 3)[0])
from seq2tensor import s2t


class Type(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(Type, self).__init__()
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(hidden_dim, int((self.hidden_dim + 7) / 2)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(int((self.hidden_dim + 7) / 2), 7)
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


id2seq_file = '../../../string/preprocessed/protein.sequences.dictionary.both.tsv'
id2index = {}
seqs = []
index = 0
for line in open(id2seq_file):
    line = line.strip().split('\t')
    id2index[line[0]] = index
    seqs.append(line[1])
    index += 1
seq_array = []
id2_aid = {}
sid = 0

seq_size = 2000
emb_files = ['../../../embeddings/default_onehot.txt', '../../../embeddings/string_vec5.txt',
             '../../../embeddings/CTCoding_onehot.txt', '../../../embeddings/vec5_CTC.txt']
use_emb = 3
hidden_dim = 50
epochs = 300

# ds_file, label_index, rst_file, use_emb, hiddem_dim
ds_file = '../../../string/preprocessed/protein.actions.SHS27k.tsv'
label_index = 4
rst_file = 'results/15k_onehot_cnn.txt'
sid1_index = 2
sid2_index = 3
if len(sys.argv) > 1:
    ds_file, label_index, rst_file, use_emb, hidden_dim, n_epochs = sys.argv[1:]
    label_index = int(label_index)
    use_emb = int(use_emb)
    hidden_dim = int(hidden_dim)
    epochs = int(n_epochs)

seq2t = s2t(emb_files[use_emb])

max_data = -1
limit_data = max_data > 0
raw_data = []
skip_head = True
x = None
count = 0

for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').split('\t')
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

class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5, 'expression': 6}
class_labels = np.zeros(len(raw_data))
for i in range(len(raw_data)):
    class_labels[i] = class_map[raw_data[i][4]]

kf = ShuffleSplit(n_splits=10)
train_test = []
tries = 10
cur = 0
for train, test in kf.split(class_labels):
    train_test.append((train, test))
    cur += 1
    if cur >= tries:
        break

# Eval params
num_total = 0
num_hit = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = nn.CrossEntropyLoss()
loss_func.to(device)
batch_size = 64
total_train_step = 0
fold = 0
# KFold
for train, test in train_test:
    type_model = None
    type_model = Type(13, hidden_dim).to(device)
    optimizer = torch.optim.Adam(type_model.parameters(), lr=0.001, eps=1e-6, amsgrad=True)
    fold += 1
    print("Fold {}:".format(fold))
    x1_data = seq_tensor[seq_index1[train]]
    x2_data = seq_tensor[seq_index2[train]]
    label_data = class_labels[train].astype(np.int64)
    train_dataset = MyDataSet(x1_data, x2_data, label_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    # Epoch
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        type_model.train()
        for x1, x2, labels in train_dataloader:
            x1, x2, labels = x1.float().to(device), x2.float().to(device), labels.to(device)
            output = type_model(x1, x2)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
            # total_train_step += 1
            # if total_train_step % 20 == 0:
            #     print("训练次数：{}, loss: {}".format(total_train_step, loss.item()))
    type_model.eval()
    with torch.no_grad():
        fold_total = 0
        fold_hit = 0
        x1_test = torch.tensor(seq_tensor[seq_index1[test]]).float().to(device)
        x2_test = torch.tensor(seq_tensor[seq_index2[test]]).float().to(device)
        label_data = class_labels[test].astype(np.int32)
        output = type_model(x1_test, x2_test)
        output = output.to('cpu')
        for i in range(len(output)):
            fold_total += 1
            if np.argmax(output[i]) == np.argmax(label_data[i]):
                fold_hit += 1

        num_total += fold_total
        num_hit += fold_hit

        accuracy = fold_hit / fold_total if fold_total > 0 else 1
        print("Accuracy:{}".format(accuracy))

accuracy = num_hit / num_total if num_total > 0 else 1
print("Accuracy:{}".format(accuracy))

with open(rst_file, 'w') as fp:
    fp.write('acc=' + str(accuracy))
