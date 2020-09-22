import jieba
import config
import numpy as np
import torch
from utils import alignWord2Char

# jieba.enable_paddle()
jieba.initialize()

jieba.load_userdict(config.vocab_path)

'''
Returns:
    vocab_size
    word2id
    id2word
'''

def get_Map_word_id():
    with open(config.vocab_path,'r') as f:
        lines=f.readlines()
    word2id={ word.strip():i for i,word in enumerate(lines)}
    id2word={ i:word.strip() for i,word in enumerate(lines)}
    return len(lines),word2id,id2word

def get_Map_char_id():
    with open(config.char_path,'r') as f:
        lines=f.readlines()
    char2id={ word.strip():i for i,word in enumerate(lines)}
    id2char={ i:word.strip() for i,word in enumerate(lines)}
    return len(lines),char2id,id2char


def tokenize(sentence):
    return jieba.lcut(sentence,HMM=False,cut_all=False)


'''
Returns:
    input_ids: batch_size * max_seq_length
    attention_mask : padding mask
'''
def sent2id(batch_sentence,word2id):
    ans=[]
    UNK=word2id.get('[UNK]')
    PAD=word2id.get('[PAD]')
    for word_list in batch_sentence:
        id_list=[word2id.get(i,UNK) for i in word_list]
        a=np.array(id_list)
        ans.append(id_list)
    input_ids,attention_mask=seq_padding(ans,padding=PAD)

    return input_ids,attention_mask
    

def seq_padding(batch_sentence,padding):
    len_lists=[ len(i) for i in batch_sentence]
    max_length=max(len_lists)

    input_ids=np.array([
        np.concatenate([x,[padding]*(max_length-(len(x)))]) if len(x)<max_length else x for x in batch_sentence
    ])

    attention_mask=np.where(input_ids!=padding,1,0)

    return input_ids,attention_mask
