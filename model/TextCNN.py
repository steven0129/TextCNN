import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class TextCNN(BasicModule):
    def __init__(self, word_dim, max_length, label_num):
        super(TextCNN, self).__init__()
        self.label_num = label_num
        self.word_dim = word_dim

        self.conv3 = nn.Conv2d(1, 1, (3, self.word_dim))
        self.conv4 = nn.Conv2d(1, 1, (4, self.word_dim))
        self.conv5 = nn.Conv2d(1, 1, (5, self.word_dim))
        self.Max3_pool = nn.MaxPool2d((max_length-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((max_length-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((max_length-5+1, 1))
        self.linear1 = nn.Linear(3, self.label_num)

    def forward(self, x):
        x = autograd.Variable(x)  # (batch_size, max_length, word_dim)
        x = x.unsqueeze(1)        # (batch_size, 1, max_length, word_dim)
        batch = x.shape[0]

        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, self.label_num)

        return x


if __name__ == '__main__':
    print('running the TextCNN...')