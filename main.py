import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import fire
from model import TextCNN
from data import TextDataset
from config import Env
from tqdm import tqdm

torch.manual_seed(1)
options = Env()
criteration = nn.CrossEntropyLoss()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    training_set = TextDataset(path='data/train/train.csv', model='wordvec/skipgram.bin', max_length=options.max_length, word_dim=options.word_dim)
    training_loader = Data.DataLoader(dataset=training_set, batch_size=options.batch_size, shuffle=True, drop_last=True)
    model = TextCNN(options.word_dim, options.max_length, training_set.encoder.classes_.shape[0])

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)

    for epoch in tqdm(range(options.epochs)):
        loss_sum = 0
        
        for data, label in tqdm(training_loader):
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()

            out = model(data)
            
            loss = criteration(out, autograd.Variable(label.squeeze().long()))
            loss_sum += loss.item() / options.batch_size
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        tqdm.write(f'epoch {epoch + 1}: loss = {loss_sum/len(training_set.data)}')
        model.save(f'checkpoints/loss-{loss_sum/len(training_set.data)}.pt')

if __name__ == '__main__':
    fire.Fire()