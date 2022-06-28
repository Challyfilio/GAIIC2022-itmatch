import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNN_Text(nn.Module):

    def __init__(self, embed_num, class_num=14, dropout=0.25):
        super(CNN_Text, self).__init__()
        embed_dim = 128
        Ci = 1
        kernel_num = 100
        Ks = [3, 4, 5]

        self.embed = nn.Embedding(embed_num, embed_dim)  # 词嵌入
        self.convs = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embed_dim)) for K in Ks])
        self.dropout = nn.Dropout(dropout)

        self.clip_model, self.preprocess = clip.load('ViT-B/32', device='cuda')
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
            nn.Linear(768, 512),
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
        x.resize_(x.shape[0], 77)
        print(x.shape)
        x = self.clip_model.encode_text(x)
        print(x.shape)  # 2,512

        '''
        x.resize_(x.shape[0], 77)
        # print(x.shape)
        x = self.clip_model.encode_text(x)
        # print(x.shape)  # 2,512
        # print('xxxxxxxxx')
        x_img = self.img_head(frt)
        # print(x_img.shape)

        # exit()
        # print(x.shape)
        # print(x.shape[0])

        # exit()
        # x = self.embed(x)  # (N, W, D)-batch,单词数量，维度
        # print(x.shape)
        # x = x.unsqueeze(1)  # (N, Ci, W, D)
        # print(x.shape)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # x = torch.cat(x, 1)
        # print(x.shape)
        # exit()



        x = torch.cat([x, x_img], axis=1)
        # print(x.shape)

        #
        logit = self.classify(x)  # (N, C)
        '''
        return logit


if __name__ == "__main__":
    net = CNN_Text(embed_num=1000)
    # net.cuda()

    text = torch.LongTensor([[1, 2, 4, 5, 2, 35, 43, 113, 111, 451, 455, 22, 45, 55],
                          [14, 3, 12, 9, 13, 4, 51, 45, 53, 17, 57, 954, 156, 23]])
    # x.cuda()
    # x.to(device)
    frt = torch.randn(2, 2048)
    # frt.cuda()
    logit = net(text, frt)
    print(logit.shape)
