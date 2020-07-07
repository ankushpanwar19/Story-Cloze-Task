import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from scipy.spatial.distance import cosine
from tqdm import tqdm

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

        sample = {'story_emb': story_emb, 'ending1_emb': ending1_emb, 'ending2_emb': ending2_emb, 'labels': labels}

        return sample

class CommonSenseData(Dataset):
    def __init__(self, data_df, embedding, device):
        
        self.data_df = data_df
        # self.transform = transform
        self.device=device
        self.embedding = embedding

        self.stemmer = PorterStemmer()
        self.emb_words = embedding.index

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        story_item = self.data_df.iloc[idx]
        story = story_item.iloc[1:-1]

        ending1_feature = self.create_common_sense_distance('ending1', story)
        ending2_feature = self.create_common_sense_distance('ending2', story)

        label = story_item['answer']-1

        sample = {
            'ending1': torch.tensor(ending1_feature,device=self.device),
            'ending2': torch.tensor(ending2_feature,device=self.device),
            'labels': torch.tensor(label,device=self.device),
        }

        return sample

    def create_common_sense_distance(self, ending_name, story):
        # row = story.iloc[idx,1:-1]
        words_e = word_tokenize(story[ending_name])[:-1]
        dist = []
        for i in tqdm(range(4)):
            dis_j = 0
            num = 0
            words_s = word_tokenize(story.iloc[i])[:-1]

            for word_e in words_e:
                max_d = 0
                num += 1
                # cnt = 0
                word_e_process = ttu.standardized_uri('en', word_e)
                if word_e_process in self.emb_words:
                    word_e_emb = self.embedding.loc[word_e_process].values
                    for word_s in words_s:
                        if self.stemmer.stem(word_e) != self.stemmer.stem(word_s):
                            word_s_process = ttu.standardized_uri('en', word_s)
                            if word_s_process in self.emb_words:
                                word_s_emb = self.embedding.loc[word_s_process].values

                                d = cosine(word_e_emb, word_s_emb)
                                if d > max_d:
                                    max_d=d
                dis_j += max_d
            dis_j /= num
            dist.append(dis_j)

        return dist


#%%
if __name__ == "__main__":
    stories_val = utils.read_data('data/nlp2_val.csv')
    embedding = pd.read_csv('numberbatch-en-19.08.txt', sep=' ', skiprows=1, header=None)
    embedding.set_index(0, inplace=True)


    # stories_val = stories_val.rename(index=str, columns=columns_rename)

    roc_dataset = CommonSenseData(stories_val, embedding, device='cpu')

    dataloader = DataLoader(roc_dataset, batch_size=4,
                            shuffle=True)

    for batch in dataloader:
        print(batch)
        break
