import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from scipy.spatial.distance import cosine
from tqdm import tqdm
from glob import glob
import os
from collections import defaultdict

import text_to_uri as ttu
import utils

nltk.download('vader_lexicon')

class RocData(Dataset):
    def __init__(self, data_df, device):
        
        self.data_df = data_df
        # self.transform = transform
        self.device=device

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        story_item = self.data_df.iloc[idx]
        full_story = ' '.join(story_item.iloc[1:5].tolist())
        ending1 = story_item['ending1']
        ending2 = story_item['ending2']
        label = story_item['answer']-1

        sample = {
            'full_story': full_story,
            'ending1': ending1,
            'ending2': ending2,
            'labels': torch.tensor(label,device=self.device),
        }

        return sample

class SentimentData(Dataset):
    def __init__(self, data_df, device, is_base_train=True):
        
        self.data_df = data_df
        self.device=device
        self.is_base_train = is_base_train

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        story_item = self.data_df.iloc[idx]
        if self.is_base_train:
            story = story_item.iloc[2:7].to_numpy()
        else:
            story = story_item.iloc[1:-1].to_numpy()

        sid = SentimentIntensityAnalyzer()
        story_senti_emb=[]
        for i in range(story.shape[0]):
            ss = sid.polarity_scores(story[i])
            story_senti_emb.append(torch.tensor([ss['neg'],ss['neu'],ss['pos']],device=self.device))

        story_emb = torch.stack(story_senti_emb[:4]).to(self.device)
        if self.is_base_train:
            label_emb = story_senti_emb[4]
            return story_emb,label_emb
        
        ending1_emb = story_senti_emb[4]
        ending2_emb = story_senti_emb[5]

        labels = story_item['answer']-1
        labels = torch.tensor(labels,device=self.device)

        sample = {
            'story_senti_emb': story_emb, 
            'ending1_senti_emb': ending1_emb, 
            'ending2_senti_emb': ending2_emb, 
            'labels': labels
        }

        return sample

class CommonSenseData(Dataset):
    def __init__(self, data_df, embedding, device):
        
        self.data_df = data_df
        # self.transform = transform
        self.device=device
        self.embedding = embedding


    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        story_item = self.data_df.iloc[idx]
        story = story_item.iloc[1:-1]
        story_id=story_item['storyid']
        ending1_feature = self.embedding[story_id]['ending1']
        ending2_feature = self.embedding[story_id]['ending2']

        label = story_item['answer']-1

        sample = {
            'ending1_common_sense': torch.tensor(ending1_feature, dtype=torch.float32, device=self.device),
            'ending2_common_sense': torch.tensor(ending2_feature, dtype=torch.float32, device=self.device),
            'labels': torch.tensor(label,device=self.device),
        }

        return sample

class CombinedData(Dataset):
    def __init__(self, data_df, cnet_embedding, device):
        self.data_df = data_df
        self.embedding = cnet_embedding
        self.device = device

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        story_item = self.data_df.iloc[idx]
        full_story = ' '.join(story_item.iloc[1:5].tolist())
        ending1 = story_item['ending1']
        ending2 = story_item['ending2']
        labels = story_item['answer']-1


        story = story_item.iloc[1:-1].tolist()
        sid = SentimentIntensityAnalyzer()
        senti_emb=[]
        for i in range(len(story)):
            ss = sid.polarity_scores(story[i])
            senti_emb.append(torch.tensor([ss['neg'],ss['neu'],ss['pos']],device=self.device))

        story_senti_emb = torch.stack(senti_emb[:4]).to(self.device)
        ending1_senti_emb = senti_emb[4]
        ending2_senti_emb = senti_emb[5]

        story_id=story_item['storyid']
        ending1_feature = self.embedding[story_id]['ending1']
        ending2_feature = self.embedding[story_id]['ending2']

        sample = {
            'full_story': full_story,
            'ending1': ending1,
            'ending2': ending2,
            'story_senti_emb': story_senti_emb, 
            'ending1_senti_emb': ending1_senti_emb, 
            'ending2_senti_emb': ending2_senti_emb,
            'ending1_common_sense': torch.tensor(ending1_feature, dtype=torch.float32, device=self.device),
            'ending2_common_sense': torch.tensor(ending2_feature, dtype=torch.float32, device=self.device),
            'labels': torch.tensor(labels,device=self.device),
        }
        return sample

class ImdbData(Dataset):
    def __init__(self, data_dir, mode, device):
        self.device = device
        self.data_dir = os.path.join(data_dir, mode)
        neg_reviews_df = self.load_reviews('neg')
        pos_reviews_df = self.load_reviews('pos')
        self.review_df = pd.concat([neg_reviews_df, pos_reviews_df])
        self.review_df = self.review_df.sample(frac=1, replace=False).reset_index(drop=True)


    def load_reviews(self, review_type):
        txt_files = glob(os.path.join(self.data_dir, review_type, '*.txt'))
        label = 0 if review_type == 'neg' else 1
        reviews = defaultdict(list)
        for fpath in txt_files:
            with open(fpath, 'r') as f:
                reviews['text'].append(f.readline())
                reviews['label'].append(label)
        reviews_df = pd.DataFrame(reviews)
        return reviews_df

    def __len__(self):
        return len(self.review_df)

    def __getitem__(self, idx):
        review_item = self.review_df.iloc[idx]
        label = review_item['label']

        sample = {
            'review': review_item['text'],
            'review_label': torch.tensor(label,dtype=torch.float32, device=self.device),
        }

        return sample


#%%
if __name__ == "__main__":
    # stories_val = utils.read_data('data/nlp2_val.csv')
    # embedding = pd.read_csv('numberbatch-en-19.08.txt', sep=' ', skiprows=1, header=None)
    # embedding.set_index(0, inplace=True)

    imdb_dataset = ImdbData('data/imdb', mode='train')



    # stories_val = stories_val.rename(index=str, columns=columns_rename)

    # roc_dataset = CommonSenseData(stories_val, embedding, device='cpu')

    dataloader = DataLoader(imdb_dataset, batch_size=4,
                            shuffle=True)

    for batch in dataloader:
        print(batch)
        break
