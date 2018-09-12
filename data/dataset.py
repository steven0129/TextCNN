from torch.utils import data
from gensim.models.wrappers import FastText
from sklearn import preprocessing
import os
import csv
import torch
import numpy as np

class TextDataset(data.Dataset):
    def __init__(self, path, model, max_length, word_dim):
        filePath = open(path)
        self.model = FastText.load_fasttext_format(model)
        self.data = list(csv.reader(filePath))
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(list(map(lambda x: x[1], self.data)))
        self.max_length = max_length
        self.word_dim = word_dim

    def __getitem__(self, index):
        texts = self.data[index][0].split(' ')
        label = self.encoder.transform([self.data[index][1]])
        vecs = list(map(self.__w2v, texts))
        if self.max_length > len(vecs):
            vecs.extend([[0] * self.word_dim] * (self.max_length - len(vecs)))
        else:
            vecs = vecs[:self.max_length]

        return torch.Tensor(vecs), torch.Tensor(label)

    def __len__(self):
        return len(self.data)

    def __w2v(self, text):
        try:
            return self.model.wv[text].tolist()
        except:
            return [0] * self.word_dim