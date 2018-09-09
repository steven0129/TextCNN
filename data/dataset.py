from torch.utils import data
from gensim.models.wrappers import FastText
from sklearn import preprocessing
import os
import csv
import torch
import numpy as np

class TextDataset(data.Dataset):
    def __init__(self, path, model):
        filePath = open(path)
        self.model = FastText.load_fasttext_format(model)
        self.data = list(csv.reader(filePath))
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(list(map(lambda x: x[1], self.data)))
        self.max_length = 200

    def __getitem__(self, index):
        texts = self.data[index][0].split(' ')
        label = self.encoder.transform([self.data[index][1]])
        vecs = list(map(self.__w2v, texts))
        vecs.extend([[0] * 300] * (self.max_length - len(vecs)))

        return torch.Tensor(vecs), torch.Tensor(label)

    def __len__(self):
        return len(self.data)

    def __w2v(self, text):
        try:
            return self.model.wv[text].tolist()
        except:
            return [0] * 300