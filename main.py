# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from config import Config
from model import TextCNN
from data import TextDataset
import argparse

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--out_channel', type=int, default=2)
parser.add_argument('--label_num', type=int, default=2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

# Create the configuration
config = Config(sentence_max_size=50,
                batch_size=args.batch_size,
                word_num=11000,
                label_num=args.label_num,
                learning_rate=args.lr,
                cuda=args.gpu,
                epoch=args.epoch,
                out_channel=args.out_channel)

training_set = TextDataset(path='data/train/train.csv', model='wordvec/skipgram.bin')
training_loader = data.DataLoader(dataset=training_set, batch_size=config.batch_size, drop_last=True)