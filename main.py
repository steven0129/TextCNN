# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from model import TextCNN
from data import TextDataset
import argparse

torch.manual_seed(1)
training_set = TextDataset(path='data/train/train.csv', model='wordvec/skipgram.bin')
training_loader = data.DataLoader(dataset=training_set, batch_size=16, shuffle=True, drop_last=True)
data, label = next(iter(training_loader))
print(data.size())