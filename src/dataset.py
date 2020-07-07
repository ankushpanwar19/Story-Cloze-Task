import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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

#%%
if __name__ == "__main__":
    stories_val = pd.read_csv('data/nlp2_val.csv')

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
    stories_val = stories_val.rename(index=str, columns=columns_rename)

    roc_dataset = RocData(stories_val)

    dataloader = DataLoader(roc_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for batch in dataloader:
        print(batch)
        break
