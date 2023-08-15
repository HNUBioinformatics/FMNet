#-- coding:utf8 --
import sys
import math
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score,f1_score, precision_score, recall_score, accuracy_score,precision_recall_curve, average_precision_score
import random
import gzip
import pickle
import timeit
import argparse
# from seq_motifs import get_motif


# GPU or CPU
if torch.cuda.is_available():
        cuda = True
        #torch.cuda.set_device(1)
        print('===> Using GPU')
else:
        cuda = False
        print('===> Using CPU')

# Filling subsequences with base nucleotide N
def padding_sequence_new(seq, window_size = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < window_size:
        gap_len = window_size -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq

# Filling entire sequence with base nucleotide N
def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

# Convert the sequence into a one-hot coding matrix
def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)

    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)

    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()

    return new_array

# Divide RNA sequence into multiple sub sequences with partial overlap

def split_overlap_seq(seq, window_size):
    overlap_size = 50
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1, window_size)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs

# Read sequence file
def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    return seq_list, labels

# Obtain the processed one-hot coding matrix and corresponding labels

def get_bag_data(data, channel = 7, window_size = 101):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        bag_seqs = split_overlap_seq(seq, window_size = window_size)
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)
        if num_of_ins > channel:
            start = (num_of_ins - channel)/2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) <channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                tri_fea = get_RNA_seq_concolutional_array('N'*window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags, labels


def get_bag_data_1_channel(data, max_len = 501):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = max_len)
        bag_subt = []
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags, labels

# Gather positive and negative data together

def read_data_file(posifile, negafile = None, train = True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label = 1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        # print(labels)

    data["seq"] = seqs
    data["Y"] = np.array(labels)

    return data

def get_data(posi, nega = None, channel = 7,  window_size = 101, train = True):
    data = read_data_file(posi, nega, train = train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len = window_size)

    else:
        train_bags, label = get_bag_data(data, channel = channel, window_size = window_size)

    return train_bags, label



class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        for idx, (X, y) in enumerate(train_loader):
            X_v = Variable(X)
            y_v = Variable(y)
            # print np.array(X_v).shape
            if cuda:
                X_v = X_v.cuda()
                y_v = y_v.cuda()
            self.optimizer.zero_grad()
            y_pred = self.model(X_v)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.item()) # need change to loss_list.append(loss.item()) for pytorch v0.4 or above

        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        print (X.shape)
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        self.model.train()

        for t in range(nb_epoch):
            loss = self._fit(train_loader)

    def evaluate(self, X, y, batch_size=32):

        y_pred = self.predict(X)
        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        if cuda:
            y_v = y_v.cuda()
        loss = self.loss_f(y_pred, y_v)
        predict = y_pred.data.cpu().numpy()[:, 1].flatten()
        auc = roc_auc_score(y, predict)
        return loss.item(), auc

    def _accuracy(self, y_pred, y):
        accuracy = float(sum(y_pred == y)) / y.shape[0]
        print(accuracy)
        return accuracy

    def predict(self, X):
        X = Variable(torch.from_numpy(X.astype(np.float32)))
        if cuda:
            X= X.cuda()
        y_pred = self.model(X)
        return y_pred

    def predict_proba(self, X):
        self.model.eval()
        return self.model.predict_proba(X)


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

    def forward(self, x):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, v)

        return attn_output



class RSBU_CW(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1, 1))
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * RSBU_CW.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * RSBU_CW.expansion),
            self.shrinkage
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != RSBU_CW.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * RSBU_CW.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * RSBU_CW.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class CNN(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=200, stride=(1, 1), padding=0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)
        out1_size = int((window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool_size = int((out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size=(1, 10), stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out2_size = int((maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool2_size = int((out2_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(maxpool2_size * nb_filter, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


class DRSN(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=256, stride=(1, 1), padding=0):
        super(DRSN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out1_size = int((window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool_size = int((out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.reslayer1 = self._make_layer(RSBU_CW, 16, 32,1, 1)
        self.reslayer2 = self._make_layer(RSBU_CW, 32, 32,1, 1)
        self.drop1 = nn.Dropout(p=0.25)
        print('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(maxpool_size * 32, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def _make_layer(self, RSBU_CW, int_channels,out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(RSBU_CW(int_channels, out_channels, stride))
            self.in_channels = out_channels * RSBU_CW.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.reslayer1(out)
        out = self.reslayer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)

        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


class CNN_LSTM(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=200, stride=(1, 1), padding=0, num_layers=2):
        super(CNN_LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)
        out1_size = int((window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool_size = int((out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.downsample = nn.Conv2d(nb_filter, 1, kernel_size=(1, 10), stride=stride, padding=padding)
        input_size = int((maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1]) + 1
        self.layer2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.reslayer21 = SelfAttention(400,400)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.downsample(out)
        out = torch.squeeze(out, 1)
        out, _ = self.layer2(out)
        out = self.reslayer21(out)
        out = out[:, -1, :]
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


class CNN_BiGRU(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=200, stride=(1, 1), padding=0, num_layers=2):
        super(CNN_BiGRU, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)
        out1_size = int((window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1)
        maxpool_size = int((out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.downsample = nn.Conv2d(nb_filter, 1, kernel_size=(1, 10), stride=stride, padding=padding)
        input_size = int((maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1]) + 1
        self.layer2 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.reslayer21 = SelfAttention(400,400)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.downsample(out)
        out = torch.squeeze(out, 1)
        out, _ = self.layer2(out)
        out = self.reslayer21(out)
        out = out[:, -1, :]
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, nb_filter = 16, kernel_size = (1, 3), stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = convR(in_channel, nb_filter, kernel_size = kernel_size, stride = stride)
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convR(nb_filter, nb_filter, kernel_size = kernel_size, stride = stride)
        self.bn2 = nn.BatchNorm2d(nb_filter)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def convR(in_channels, out_channels, kernel_size, stride=1, padding = (0, 1)):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, stride=stride, bias=False)



class ResNet(nn.Module):
    def __init__(self, block, layers, nb_filter = 16, channel = 7, labcounts = 12, window_size = 36, kernel_size = (1, 3), pool_size = (1, 3), num_classes=2, hidden_size = 200):
        super(ResNet, self).__init__()
        self.in_channels = channel
        self.conv = convR(self.in_channels, nb_filter, kernel_size = (4, 10))
        cnn1_size = window_size - 7
        self.bn = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, nb_filter, layers[0],  kernel_size = kernel_size)
        self.layer2 = self.make_layer(block, nb_filter*2, layers[1], 1, kernel_size = kernel_size, in_channels = nb_filter)
        self.avg_pool = nn.AvgPool2d(pool_size)
        avgpool2_1_size = int((cnn1_size - (pool_size[1] - 1) - 1)/pool_size[1]) + 1
        last_layer_size = 2*nb_filter*avgpool2_1_size
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1,  kernel_size = (1, 10), in_channels = 16):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                convR(in_channels, out_channels, kernel_size = kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, kernel_size = kernel_size, stride = stride, downsample = downsample))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size = kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        if cuda:
            x = x.cuda()
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


def train_network(model_type, X_train, y_train, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16, motif = False, motif_seqs = [], motif_outdir = 'motifs'):
    print ('model training for ', model_type)
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNDRSN':
        model =  DRSN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'CNNBiGRU':
        model =CNN_BiGRU(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'CNNResNet':
        model = ResNet(ResidualBlock, [3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    else:
        print('Please enter the required model')

    if cuda:
            model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001),
                loss=nn.CrossEntropyLoss())
    clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs)
    if motif and channel == 1:
        detect_motifs(model, motif_seqs, X_train, motif_outdir)

    torch.save(model.state_dict(), model_file)


def predict_network(model_type, X_test, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16):
    print ('model training for ', model_type)

    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNDRSN':
        model =  DRSN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'CNNBiGRU':
        model =CNN_BiGRU(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'CNNResNet':
        model = ResNet(ResidualBlock, [3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)

    else:
        print('Please enter the required model')

    if cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(model_file))
    try:
        pred = model.predict_proba(X_test)
    except: #to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis = 0)
    return pred


def batch(tensor, batch_size = 1000):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

def detect_motifs(model, test_seqs, X_train, output_dir = 'motifs', channel = 1):
    if channel == 1:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for param in model.parameters():
            layer1_para =  param.data.cpu().numpy()
            break
        N = len(test_seqs)
        if N > 15000: # do need all sequence to generate motifs and avoid out-of-memory
        	sele = 15000
        else:
        	sele = N
        ix_all = np.arange(N)
        np.random.shuffle(ix_all)
        ix_test = ix_all[0:sele]

        X_train = X_train[ix_test, :, :, :]
        test_seq = []
        for ind in ix_test:
        	test_seq.append(test_seqs[ind])
        test_seqs = test_seq
        filter_outs = model.layer1out(X_train)[:,:, 0, :]
        get_motif(layer1_para[:,0, :, :], filter_outs, test_seqs, dir1 = output_dir)


def Indicators(y, y_pre):
    '''
    :param y: array，True value
    :param y_pre: array，Predicted value
    :return: float
    '''
    lenall = len(y)
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(lenall):
        if y_pre[i] == 1:
            if y[i] == 1:
                TP += 1
            if y[i] == 0:
                FP += 1
        if y_pre[i] == 0:
            if y[i] == 1:
                FN += 1
            if y[i] == 0:
                TN += 1
    member = TP * TN - FP * FN
    mcc = float(TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP)))
    acc = float((TP + TN) / (TP + TN + FP + FN))
    recall = float(TP / (TP + FN))
    pre = float(TP / (TP + FP))
    f1 = float(2 * pre * recall / (pre + recall))
    # demember = ((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)) ** 0.5
    # mcc = member / demember
    return mcc, acc, recall, pre, f1


def run(parser):
    posi = parser.posi
    nega = parser.nega
    out_file = parser.out_file
    train = parser.train
    model_file = parser.model_file
    predict = parser.predict
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    start_time = timeit.default_timer()
    motif = parser.motif
    motif_outdir = parser.motif_dir

    #pdb.set_trace()
    if predict:

        train = False
        if testfile == '':
            print ('you need specify the fasta file for predicting when predict is True')
            return
    if train:
        if posi == '' or nega == '':
            print ('you need specify the training positive and negative fasta file for training when train is True')
            return

    if train:
        motif_seqs = []
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']

        print("1011")
        train_bags, train_labels = get_data(posi, nega, channel = 7, window_size = 101)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_type + '.1011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_type + '.1011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNDRSN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_type + '.1011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNBiGRU"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7, window_size=101 + 6,model_file=model_type + '.1011', batch_size=batch_size, n_epochs=n_epochs, num_filters=num_filters)
        model_type = "CNNResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel=7, window_size=101 + 6, model_file=model_type + '.1011', batch_size=batch_size, n_epochs=n_epochs, num_filters=num_filters)


        print("2011")
        train_bags, train_labels = get_data(posi, nega, channel = 3, window_size = 201)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type =  "CNNDRSN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNBiGRU"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type ="CNNResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)


        print("3011")
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 301)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_type + '.3011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_type + '.3011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type =  "CNNDRSN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_type + '.3011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNBiGRU"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2, window_size=301 + 6, model_file=model_type + '.3011', batch_size=batch_size, n_epochs=n_epochs, num_filters=num_filters)
        model_type = "CNNResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel=2, window_size=301 + 6, model_file=model_type + '.3011', batch_size=batch_size, n_epochs=n_epochs, num_filters=num_filters)


        print("4011")
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 401)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_type + '.4011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_type + '.4011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type =  "CNNDRSN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_type + '.4011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNBiGRU"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size=401 + 6, model_file=model_type + '.4011', batch_size=batch_size, n_epochs=n_epochs, num_filters=num_filters)
        model_type = "CNNResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size=401 + 6, model_file=model_type + '.4011', batch_size=batch_size, n_epochs=n_epochs, num_filters=num_filters)


    elif predict:

        fw = open(out_file, 'w')
        file_out= open('result.txt', 'a')

        model_type = "CNN"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        CnnPre11 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.1011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        CnnPre21 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        CnnPre31 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.3011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        CnnPre41 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.4011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)


        model_type = "CNNLSTM"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        CnnLstmPre11 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.1011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        CnnLstmPre21  = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        CnnLstmPre31 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.3011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        CnnLstmPre41 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.4011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)


        model_type = "CNNDRSN"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        DRSNPre11 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.1011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        DRSNPre21 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        DRSNPre31 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.3011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        DRSNPre41 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.4011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)


        model_type = "CNNBiGRU"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        BiGRUPre11 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.1011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        BiGRUPre21 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        BiGRUPre31 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.3011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        BiGRUPre41 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.4011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)


        model_type = "CNNResNet"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        ResNetPre11 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.1011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        ResNetPre21 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.2011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        ResNetPre31 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.3011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        ResNetPre41 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.4011', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)


        CnnPre1 = (1*CnnPre11+2*CnnPre21+4*CnnPre31+3*CnnPre41)/10.0
        CnnLstmPre1 = (1*CnnLstmPre11+ 2*CnnLstmPre21+ 4*CnnLstmPre31+ 3*CnnLstmPre41)/10.0
        DRSNPre1= (1*DRSNPre11+2*DRSNPre21+4*DRSNPre31+3*DRSNPre41)/10.0
        BiGRUPre1= (1*BiGRUPre11+2*BiGRUPre21+4*BiGRUPre31+3*BiGRUPre41)/10.0
        ResNetPre1= (1*ResNetPre11+2*ResNetPre21+4*ResNetPre31+3*ResNetPre41)/10.0
        predict_sum= (4*CnnPre1+2*CnnLstmPre1+5*DRSNPre1+1*BiGRUPre1+3*ResNetPre1)/15.0

        p = predict_sum
        p = np.around(p, 0).astype(int)
        mcc, accuracy, recall, precision, f1 = Indicators(X_labels, p)
        ap = average_precision_score(X_labels, p)
        print('accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'mcc:', mcc, 'f1:', f1, 'ap:', ap)
        auc = roc_auc_score(X_labels, predict_sum)
        print('AUC:{:.3f}'.format(auc))
        myprob = "\n".join(map(str, predict_sum))
        fw.write(myprob)
        fw.close()
        file_out.write(str(round(float(auc), 3)) + '\n')
        file_out.close()


    elif motif:
        motif_seqs = []
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']
        if posi == '' or nega == '':
            print ('To identify motifs, you need training positive and negative sequences using CNN.')
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 501)
        train_network("CNN", np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_file + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir)
    else:
        print ('please specify that you want to train the mdoel or predict for your own sequences')


def parse_arguments(parser):
    parser.add_argument('--posi', type=str, metavar='<postive_sequecne_file>', help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='<negative_sequecne_file>', help='The fasta file of negative training samples')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='The output file used to store the prediction probability of the testing sequences')
    parser.add_argument('--train', type=bool, default=False, help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='model.pkl', help='The file to save model parameters. Use this option if you want to train on your sequences or predict for your sequences')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--testfile', type=str, default='',  help='the test fast file for sequences you want to predict for, you need specify it when using predict')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--num_filters', type=int, default=16, help='The number of filters for CNNs (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs (default value: 50)')
    parser.add_argument('--motif', type=bool, default=False, help='It is used to identify binding motifs from sequences.')
    parser.add_argument('--motif_dir', type=str, default='motifs', help='The dir used to store the prediction binding motifs.')
    parser.add_argument('--data',default='ALKBH5', type=str,  help='data')
    args = parser.parse_args()
    return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print (args)
run(args)
