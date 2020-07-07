#%%
import os
import argparse
import math
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import SentimentData
from bert_model import SentimentNet

if torch.cuda.is_available():
    device=torch.device('cuda')
    print('running on gpu')
else:
    device=torch.device('cpu')
    print('running on cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=100)

FLAGS = parser.parse_args()
NUM_EPOCHS = FLAGS.num_epochs
LR = FLAGS.lr
BATCH_SIZE = FLAGS.batch_size
PRINT_EVERY = FLAGS.print_every

train_stories = pd.read_csv('data/nlp2_train.csv')
# train_stories, val_stories = train_test_split(stories, test_size=0.2)

train_dataloader = DataLoader(SentimentData(train_stories,device), batch_size=BATCH_SIZE,
                                shuffle=True)

net = SentimentNet()
net.to(device)

criterion = torch.nn.CosineSimilarity()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

n_iteration=len(train_dataloader)
# v_iteration=len(val_dataloader)

for epoch in range(NUM_EPOCHS):
    running_loss_train = 0.0
    for i, (train_batch,label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = net(train_batch)
        loss =torch.mean(1- criterion(output,label))
        loss.backward()
        optimizer.step()

        running_loss_train +=loss.item()

        if i%PRINT_EVERY == 0:
            print(f'Epoch: {epoch+1}, Step: {i}/{n_iteration},\
                Runningloss: {running_loss_train/PRINT_EVERY}')
            running_loss_train = 0.0


print('end')


# %%
