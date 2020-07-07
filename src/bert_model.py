import torch
from transformers import BertModel


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
