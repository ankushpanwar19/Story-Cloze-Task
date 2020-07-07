import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
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
    def __init__(self, data_df, device):
        
        self.data_df = data_df
        # self.transform = transform
        self.device=device

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        story_item = self.data_df.iloc[idx]
        story = story_item.iloc[2:7].to_numpy()

        sid = SentimentIntensityAnalyzer()
        story_senti_emb=[]
        for i in range(story.shape[0]):
            ss = sid.polarity_scores(story[i])
            story_senti_emb.append(torch.tensor([ss['neg'],ss['neu'],ss['pos']],device=self.device))

        story_emb = torch.stack(story_senti_emb[:4]).to(self.device)
        label_emb = story_senti_emb[4]

        return story_emb,label_emb

#%%
if __name__ == "__main__":
    stories_val = pd.read_csv('data/nlp2_train.csv')

    # stories_val = stories_val.rename(index=str, columns=columns_rename)

    roc_dataset = SentimentData(stories_val,device='cpu')

    dataloader = DataLoader(roc_dataset, batch_size=4,
                            shuffle=True)

    for batch in dataloader:
        print(batch)
        break
