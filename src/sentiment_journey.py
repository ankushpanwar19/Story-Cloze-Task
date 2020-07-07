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
from metrics import Accuracy

import utils

if torch.cuda.is_available():
    device=torch.device('cuda')
    print('running on gpu')
else:
    device=torch.device('cpu')
    print('running on cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=4)
# parser.add_argument('--is_base_train', type=bool, default=True)

FLAGS = parser.parse_args()
NUM_EPOCHS = FLAGS.num_epochs
LR = FLAGS.lr
BATCH_SIZE = FLAGS.batch_size
PRINT_EVERY = FLAGS.print_every
# IS_BASE_TRAIN = FLAGS.is_base_train

train_stories = pd.read_csv('data/nlp2_train.csv')
val_stories = utils.read_data('data/nlp2_val.csv')
test_stories = utils.read_data('data/nlp2_test.csv')

# train_stories, val_stories = train_test_split(stories, test_size=0.2)

train_dataloader = DataLoader(SentimentData(train_stories,device), batch_size=BATCH_SIZE,
                                shuffle=True)
val_dataloader = DataLoader(SentimentData(val_stories,device, is_base_train=False), batch_size=BATCH_SIZE,
                                shuffle=False)

net = SentimentNet()
net.to(device)
metric_acc = Accuracy()

criterion = torch.nn.CosineSimilarity()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

n_iteration=len(train_dataloader)

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
        break

    with torch.no_grad():
        for i, val_batch in enumerate(val_dataloader):
            output = net(val_batch['story_emb'])

            ending1_sim = criterion(output,val_batch['ending1_emb'])
            ending2_sim = criterion(output,val_batch['ending2_emb'])

            ending_sim = torch.stack((ending1_sim, ending2_sim), dim=1)

            _, predicted = torch.max(ending_sim, 1)
            metric_acc.update_batch(predicted, val_batch['labels'])

        val_accuracy = metric_acc.get_metrics_summary()
        metric_acc.reset()
        print(f'============Epoch: {epoch+1}, ValAccuracy: {val_accuracy}=================')

torch.save(net.state_dict(), 'checkpoints/sentiment_base.pth')

print('end')


# %%
