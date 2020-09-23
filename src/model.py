import torch
import torch.nn as nn
import numpy as np

class DGCNN(nn.Module):
    def __init__(self,word_emb_size,**kwargs):
        super(DGCNN,self).__init__()
        self.h_dim=word_emb_size
        self.embedding=MixEmbedding(word_emb_size,kwargs['char_file'],kwargs['word_file'])
        
        self.question_encoder=nn.Sequential(
            DilatedGatedConv1D(self.h_dim,dilation=1),
            DilatedGatedConv1D(self.h_dim,dilation=2),
            DilatedGatedConv1D(self.h_dim,dilation=1),
            PoolingAttention(self.h_dim,self.h_dim)
            )
        self.linear=nn.Linear(self.h_dim * 2, self.h_dim,bias=False)
        self.envidence_encoder=nn.Sequential(
            DilatedGatedConv1D(self.h_dim,dilation=1),
            DilatedGatedConv1D(self.h_dim,dilation=2),
            DilatedGatedConv1D(self.h_dim,dilation=4),
            DilatedGatedConv1D(self.h_dim,dilation=8),
            DilatedGatedConv1D(self.h_dim,dilation=16),
            DilatedGatedConv1D(self.h_dim,dilation=1)
        )

        self.poolAttention=PoolingAttention(self.h_dim*2,self.h_dim)

        self.context_classifier=nn.Linear(self.h_dim * 2,1)
        self.start_classfier=nn.Linear(self.h_dim * 2,1)
        self.end_classfier=nn.Linear(self.h_dim * 2,1)

        self.dropout=nn.Dropout(p=0.1)

    def forward(self,inputs):
        Qc,Qw,q_mask,Ec,Ew,e_mask=inputs
        q=self.embedding([Qw,Qc])
        q=self.dropout(q)

        e=self.embedding([Ew,Ec])
        e=self.dropout(e)
        max_seq_len_e=e.shape[1]

        qv=self.question_encoder([q,q_mask])
        #batch_size , word_emb_dim
        qv=qv.unsqueeze(1).expand(-1,max_seq_len_e,-1)
        #batch_size , max_seq_len_e , word_emb_dim

        e=torch.cat((e,qv),dim=-1)
        #concatenate qv into e 
        # e: batch_size , max_seq_len_e, word_emb_dim * 2
        e = self.linear(e)
        # e: batch_size , max_seq_len_e, word_emb_dim
        e, _= self.envidence_encoder([e,e_mask])
        # e: batch_size , max_seq_len_e ,  word_emb_dim
        eq = torch.cat((e,qv),dim=-1)
        # eq: batch_size , max_seq_len_e , word_emb_dim * 2
        
        ev=self.poolAttention([eq,e_mask])
        # ev: batch_size , word_emb_dim * 2

        # Gate
        ev1 = torch.sigmoid(self.context_classifier(ev))
        # ev1 : batch_size , 1
        ev1 = ev1.unsqueeze(1).expand(-1,max_seq_len_e,-1)
        # ev1 : batch_size, max_seq_len_e, 1

        As_ = torch.sigmoid(self.start_classfier(eq))
        Ae_ = torch.sigmoid(self.end_classfier(eq))
        # As_ , Ae_ : batch_size , max_seq_len_e , 1

        As_,Ae_ = As_ * ev1 ,Ae_ * ev1

        As_ = As_.squeeze(-1)
        Ae_ = Ae_.squeeze(-1)
        # As_ , Ae_ : batch_size , max_seq_len_e
        return As_ , Ae_
    


    
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
        # import pdb;pdb.set_trace()
        x,attention_mask=inputs
        xo=self.map2(self.activate(self.map(x)))
        # batch_size, max_seq_len, 1

        attention_mask=attention_mask.unsqueeze(-1)
        #set padding attention value to negative infanitive
        m=self.MAX
        xo=torch.where(attention_mask==1,xo,m.to(x.device))
        # batch_size, max_seq_len,1

        xo=torch.softmax(xo,axis=1)
        xo.expand(-1,-1,self.in_dim)
        # batch_size, max_seq_len, word_emb_dim
        x=torch.sum(xo*x,axis=1)
        # batch_size, word_emb_dim

        return x
    

class MixEmbedding(nn.Module):

    def __init__(self,in_dim,char_file,word_file):
        '''
        '''
        super(MixEmbedding,self).__init__()
        # self.char_nums=char_nums
        self.emb_dim=in_dim
        #字向量 预训练参数加载
        self.char_embedding=nn.Embedding.from_pretrained(torch.from_numpy(np.load(char_file)),padding_idx=0).float()

        #词向量加载 并冻结参数
        self.word_embedding=nn.Embedding.from_pretrained(torch.from_numpy(np.load(word_file)),padding_idx=0).float()
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


class DilatedGatedConv1D(nn.Module):
    '''
    DGCNN
    '''

    def __init__(self,h_dim,dilation,k_size=3,drop_gate=0.1):
        super(DilatedGatedConv1D,self).__init__()
        self.h_dim = h_dim
        self.dilation = dilation
        self.kernel_size = k_size
        self.dropout=nn.Dropout(p=drop_gate)
        self.padding=self.dilation *(self.kernel_size-1)//2
        #input  batch_size , Channel_in , seq_len
        self.conv1=nn.Conv1d(in_channels=self.h_dim,out_channels=self.h_dim,\
            kernel_size=self.kernel_size,dilation=dilation,padding=self.padding)
        self.conv2=nn.Conv1d(in_channels=self.h_dim,out_channels=self.h_dim,\
            kernel_size=self.kernel_size,dilation=dilation,padding=self.padding)

    def forward(self,inputs):
        x,mask=inputs
        #x batch_size , seq_max_len , word_emb_size
        #mask 部分置0 batch_size , seq_max_len
        
        x=x.permute(0,2,1).contiguous()
        # batch_size , word_emb_size , seq_max_len
        mask_=mask.unsqueeze(1).expand(-1,x.shape[1],-1)
        
        x=x*mask_
        # x,mask: batch_size , word_emb_size , seq_max_len

        x1=self.conv1(x)
        x2=self.conv2(x)
        # batch_size , word_emb_size, seq_max_len

        x2=self.dropout(x2)
        x2=torch.sigmoid(x2)

        #add resnet and multiply a gate for this resnet layer
        xx=(1-x2)* x + x2 * x1
        #batch_size , word_emb_size, seq_max_len
        xx=xx.permute(0,2,1).contiguous()
        return [xx,mask]





        
