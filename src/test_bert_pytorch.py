#%%
import os
import argparse
import math
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import RocData
from bert_model import BertNet
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
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--batch_size', type=int, default=4)

FLAGS = parser.parse_args()
NUM_EPOCHS = FLAGS.num_epochs
LR = FLAGS.lr
BATCH_SIZE = FLAGS.batch_size
PRINT_EVERY = FLAGS.print_every

#%%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

stories = pd.read_csv('data/nlp2_val.csv')

columns_rename = {
    'InputStoryid': 'storyid',
    'InputSentence1': 'sentence1',
    'InputSentence2': 'sentence2',
    'InputSentence3': 'sentence3',
    'InputSentence4': 'sentence4',
    'InputSentence5': 'sentence5',
    'RandomFifthSentenceQuiz1': 'ending1',
    'RandomFifthSentenceQuiz2': 'ending2',
    'AnswerRightEnding': 'answer'
}
stories = stories.rename(index=str, columns=columns_rename)
train_stories, val_stories = train_test_split(stories, test_size=0.1)


train_dataloader = DataLoader(RocData(train_stories), batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=4)
val_dataloader = DataLoader(RocData(val_stories), batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4)

net = BertNet()
net.to(device)

ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

metric_acc = Accuracy()

n_iteration=len(train_dataloader)
v_iteration=len(val_dataloader)

for epoch in range(NUM_EPOCHS):
    running_loss_train = 0.0
    running_loss_val = 0.0
    for i, train_batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        output, train_loss = utils.run_step(train_batch, net, tokenizer, ce_loss, device)

        train_loss.backward()
        optimizer.step()

        running_loss_train += train_loss.item()

        _, predicted = torch.max(output, 1)
        metric_acc.update_batch(predicted, train_batch['labels'])

        if i%PRINT_EVERY == 0:
            train_accuracy = metric_acc.get_metrics_summary()
            metric_acc.reset()

            print(f'Epoch: {epoch+1}, Step: {i}/{n_iteration}, Accuracy: {train_accuracy}, \
                    Runningloss: {running_loss_train/PRINT_EVERY}')
            running_loss_train = 0


    with torch.no_grad():
        for i, val_batch in enumerate(val_dataloader):
            output, val_loss = utils.run_step(val_batch, net, tokenizer, ce_loss, device)

            running_loss_val += val_loss.item()

            _, predicted = torch.max(output, 1)
            metric_acc.update_batch(predicted, val_batch['labels'])

        val_accuracy = metric_acc.get_metrics_summary()
        metric_acc.reset()

        print(f'============Epoch: {epoch+1}, ValAccuracy: {val_accuracy}, Valloss: {running_loss_val/v_iteration}=================')