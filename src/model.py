import torch
import torch.nn as nn
import numpy as np

class DGCNN(nn.Module):
    def __init__(self):
        pass
    
'''
X batch_size , max_seq_len, word_emb_dim
Attention_mask batch_size,max_seq_len
'''

class PoolingAttention(nn.Module):
    
    def __init__(self,w_emb_dim,h_dim):
        super(PoolingAttention,self).__init__()
        self.in_dim=w_emb_dim
        self.h_dim=h_dim
        self.map=nn.Linear(self.in_dim,self.h_dim,bias=False)
        self.activate=nn.Tanh()
        self.map2=nn.Linear(self.h_dim,1,bias=False)
        self.MAX=torch.from_numpy(np.array(-1e12)).float()
    def forward(self,inputs):
        x,attention_mask=inputs
        xo=self.map2(self.activate(self.map(x)))
        # batch_size, max_seq_len, 1

        attention_mask=attention_mask.unsqueeze(-1)
        #set padding attention value to negative infanitive
        xo=torch.where(attention_mask,xo,self.MAX)
        # batch_size, max_seq_len,1

        xo=torch.softmax(xo,axis=1)
        xo.expand(-1,-1,self.in_dim)
        # batch_size, max_seq_len, word_emb_dim
        x=torch.sum(xo*x,axis=1)
        # batch_size, word_emb_dim

        return x
    

class MixEmbedding(nn.Module):

    def __init__(self,char_nums,in_dim,word_file):
        '''
        '''
        super(MixEmbedding,self).__init__()
        self.char_nums=char_nums
        self.emb_dim=in_dim
        self.char_embedding=nn.Embedding(self.char_nums,self.emb_dim,padding_idx=0)
        
        #词向量加载 并冻结参数
        self.word_embedding=nn.Embedding.from_pretrained(torch.from_numpy(np.load(word_file)),padding_idx=0)
        for i in self.word_embedding.parameters():
            i.requires_grad=False

        self.word_linear=nn.Linear(self.word_embedding.embedding_dim,self.emb_dim,bias=False)
    
    def forward(self,inputs):
        '''
        word batch_size, word_max_len
        char batch_size, char_max_len
        '''
        word,char=inputs
        word=self.word_embedding(word)
        char=self.char_embedding(char)
        word=self.word_linear(word)

        return word+char




        
