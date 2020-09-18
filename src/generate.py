
import config
import numpy as np
from tokenizer import seq_padding,sent2id,tokenize
import random

'''
Only Word Embedding
No char Embedding
'''


def find_index(e,a):
    '''
    e: list of str ['今天','是','星期五']
    a: list of str ['星期五']
    Retruns:
    the index of sub str a in e
    '''
    SEP="##@@##"
    e_str=SEP.join(e)
    a_str=SEP.join(a)
    f=e_str.find(a_str)



class data_generator(object):

    def __init__(self,data,word2id,batch_size=config.batch_size):
        self.data=data
        self.batch_size=batch_size
        self.steps=len(self.data) // self.batch_size
        self.word2id=word2id
        if len(self.data) % self.batch_size !=0:
            self.steps+=1
    
    def __len__(self):
        return self.steps
    
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            Q,E,As,Ae=[],[],[],[]
            for i in idxs:
                item=self.data[i]

                #question
                q=item['question']
                e=item['evidence']
                a=item['answer']
                a=random.choice(a)

                q=tokenize(q)
                e=tokenize(e)
                a=tokenize(a)

                a1,a2=
                Q.append()

                





                



                
