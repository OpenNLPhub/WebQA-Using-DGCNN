import jieba
import config
import numpy as np
import torch
from utils import alignWord2Char
import gensim
import pickle
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
    with open(config.word_path,'rb') as f:
        word2id=pickle.load(f,encoding='utf-8')
    return len(word2id),word2id,0

def get_Map_char_id():
    with open(config.char_path,'rb') as f:
        char2id=pickle.load(f,encoding='utf-8')
    return len(char2id),char2id,0

# def word_char_id():
#     model=gensim.models.Word2Vec.load(config.wv_baidu_path)
#     vocab=model.wv.vocab
#     word2id={'[PAD]':0,'[UNK]':1}
#     char2id={'[PAD]':0,'[UNK]':1}
#     for w in tqdm(vocab)


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
    

def seq_padding(batch_sentence,padding=0):
    len_lists=[ len(i) for i in batch_sentence]
    max_length=max(len_lists)

    input_ids=np.array([
        np.concatenate([x,[padding]*(max_length-(len(x)))]) if len(x)<max_length else x for x in batch_sentence
    ])

    attention_mask=np.where(input_ids!=padding,1,0)

    return input_ids,attention_mask


if __name__ == '__main__':
    import pdb;pdb.set_trace()
    _,word2id,_=get_Map_word_id()
    _,char2id,_=get_Map_char_id()
    print(len(word2id))
    print(len(char2id))