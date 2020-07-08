import torch
from transformers import BertModel
import torch.nn.functional as F


class BertNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = torch.nn.Linear(768, 1)

    def forward(self, ending1, ending2):
        e1 = self.bert(**ending1)
        e1_score = self.out(e1[0][:,0,:])

        e2 = self.bert(**ending2)
        e2_score = self.out(e2[0][:,0,:])

        output = torch.cat((e1_score, e2_score), dim=1)

        return output

class SentimentNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(3,3,batch_first=True)

    def forward(self,data):
        output,(h,c)=self.lstm1(data)

        return h[-1]

class SentimentNetEnd2End(SentimentNet):
    def __init__(self):
        super().__init__()
        
        self.criterion=torch.nn.CosineSimilarity()

    def forward(self,data):
        output,(h,c)=self.lstm1(data['story_emb'])

        ending1_sim = self.criterion(h[-1],data['ending1_emb'])
        ending2_sim = self.criterion(h[-1],data['ending2_emb'])

        ending_sim = torch.stack((ending1_sim, ending2_sim), dim=1)

        return ending_sim

class CommonsenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = torch.nn.Linear(4, 256)
        self.out = torch.nn.Linear(256, 1)

    def forward(self,data):
        out1=self.step(data['ending1'])
        out2=self.step(data['ending2'])

        ending_prob = torch.cat((out1, out2),dim=1)

        return ending_prob

    def step(self,ending):
        out=F.relu(self.layer1(ending))
        out=self.out(out)

        return out