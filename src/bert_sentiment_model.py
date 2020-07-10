import os
import argparse
import math
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import ImdbData
from all_models import SentimentBertNet
from metrics import Accuracy

import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device=torch.device('cuda')
    print('running on gpu')
else:
    device=torch.device('cpu')
    print('running on cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--batch_size', type=int, default=32)

FLAGS = parser.parse_args()
NUM_EPOCHS = FLAGS.num_epochs
LR = FLAGS.lr
BATCH_SIZE = FLAGS.batch_size
PRINT_EVERY = FLAGS.print_every

DATA_DIR = 'data/imdb'

train_dataloader = DataLoader(ImdbData(DATA_DIR, 'train', device), batch_size=BATCH_SIZE,
                                shuffle=True)
val_dataloader = DataLoader(ImdbData(DATA_DIR, 'test', device), batch_size=BATCH_SIZE,
                                shuffle=False)


net = SentimentBertNet(device)
net.to(device)

ce_loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

metric_acc = Accuracy()

n_iteration=len(train_dataloader)
v_iteration=len(val_dataloader)
val_accuracy_prev = 0

for epoch in range(NUM_EPOCHS):
    running_loss_train = 0.0
    running_loss_val = 0.0
    for i, train_batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = net(train_batch)
        output = output.squeeze(dim=-1)
        train_loss = ce_loss(output, train_batch['review_label'])

        train_loss.backward()
        optimizer.step()

        running_loss_train += train_loss.item()

        predicted = (output > 0.5).type(torch.int)
        metric_acc.update_batch(predicted, train_batch['review_label'])

        if i%PRINT_EVERY == 0:
            train_accuracy = metric_acc.get_metrics_summary()
            metric_acc.reset()

            print(f'Epoch: {epoch+1}, Step: {i}/{n_iteration}, Accuracy: {train_accuracy}, \
                    Runningloss: {running_loss_train/PRINT_EVERY}')
            running_loss_train = 0
        break

    with torch.no_grad():
        for i, val_batch in enumerate(val_dataloader):
            output = net(val_batch)
            output = output.squeeze(dim=-1)
            val_loss = ce_loss(output, val_batch['review_label'])

            running_loss_val += val_loss.item()

            predicted = (output > 0.5).type(torch.int)
            metric_acc.update_batch(predicted, val_batch['review_label'])

        val_accuracy = metric_acc.get_metrics_summary()
        metric_acc.reset()
        if val_accuracy > val_accuracy_prev:
            torch.save(net.state_dict(), 'checkpoints/bert_sentiment.pth')
            print('checkpoint saved')
            val_accuracy_prev = val_accuracy

        print(f'============Epoch: {epoch+1}, ValAccuracy: {val_accuracy}, Valloss: {running_loss_val/v_iteration}=================')


print('end')