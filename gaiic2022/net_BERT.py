import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertConfig


# class Bert_Model(nn.Module):
#     def __init__(self, bert_path='bert_model/', classes=13):
#         super(Bert_Model, self).__init__()
#
#         self.config = BertConfig.from_pretrained(bert_path)
#         self.bert = BertModel.from_pretrained(bert_path)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#         # self.fc = nn.Linear(self.config.hidden_size, classes)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None):
#         output = self.bert(input_ids, token_type_ids, attention_mask)[1]  # 池化后的输出
#         # logit = self.fc(output)
#         return output


class CNN_Text(nn.Module):

    def __init__(self, embed_num, class_num=14, dropout=0.25, model_name='bert-base-chinese'):
        super(CNN_Text, self).__init__()
        embed_dim = 128
        Ci = 1
        kernel_num = 100
        Ks = [3, 4, 5]

        # CNN_Text
        self.embed = nn.Embedding(embed_num, embed_dim)  # 词嵌入
        self.convs = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embed_dim)) for K in Ks])

        # Bert
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(dropout)
        #
        self.img_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            # nn.Dropout(p = dropout),
            #
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            # nn.Dropout(p = dropout),
            #
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.Dropout(p = dropout),
            #
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            #
        )
        self.classify = nn.Sequential(
            #
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            #
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            #
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, class_num)
        )
        #

    def forward(self, x, frt):
        # x = self.embed(x)  # (N, W, D)-batch,单词数量，维度
        # x = x.unsqueeze(1)  # (N, Ci, W, D)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = self.bert(x, token_type_ids=None, attention_mask=None)[1]  # 池化后的输出,是向量
        # x = torch.cat(x, 1)
        x_img = self.img_head(frt)
        x = torch.cat([x, x_img], axis=1)
        #
        logit = self.classify(x)  # (N, C)
        return logit


if __name__ == "__main__":
    net = CNN_Text(embed_num=1000)
    x = torch.LongTensor([[1, 2, 4, 5, 2, 35, 43, 113, 111, 451, 455, 22, 45, 55],
                          [14, 3, 12, 9, 13, 4, 51, 45, 53, 17, 57, 954, 156, 23]])

    frt = torch.randn(2, 2048)
    logit = net(x, frt)
    print(logit.shape)

    # model = Bert_Model()
    # output = model(x)
    # print(output.shape)
