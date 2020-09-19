
import config
import numpy as np
from tokenizer import seq_padding,sent2id,tokenize
import random
import re
'''
Only Word Embedding
No char Embedding
'''


# def find_index(e,a):
#     '''
#     e: list of str ['今天','是','星期五']
#     a: list of str ['星期五']
#     Retruns:
#     the index of sub str a in e
#     '''
#     SEP="##@@##"
#     e_str=SEP.join(e)
#     a_str=SEP.join(a)
#     f=e_str.find(a_str)
#     i=0
#     head=0
#     while i<=f:
#         if e_str[i:i+len(SEP)]==SEP:




class data_generator(object):
    def __init__(self,data,word2id,char2id,batch_size=config.batch_size):
        self.data=data
        self.batch_size=batch_size
        self.steps=len(self.data) // self.batch_size
        self.word2id=word2id
        self.char2id=char2id
        if len(self.data) % self.batch_size !=0:
            self.steps+=1
    
    def __len__(self):
        return self.steps
    
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            Qc,Qw,Ec,Ew,As,Ae=[],[],[],[],[],[]
            for i in idxs:
                item=self.data[i]

                #question
                q=item['question']
                q_char=list(q)
                q_word=tokenize(q)
                
                #evidence
                e=item['evidence']
                e_char=list(q)
                e_word=tokenize(e)

                #answer
                a=item['answer']
                a=random.choice(a)
                a_char=list(a)
                
                a1,a2 =  np.zeros(len(e_char)) , np.zeros(len(e_char))

                for j in re.finditer(re.escape(a),e):
                    a1[j.start()]=1
                    a2[j.end()-1]=1
                Qc.append(q_char)
                Qw.append(q_word)
                Ec.append(e_char)
                Ew.append(e_word)
                As.append(a1)
                Ae.append(a2)

                if len(Qc)==self.batch_size or i==idxs[-1]:
                    Qc=sent2id(Qc,self.char2id)
                    Qw=sent2id(Qw,self.word2id)

                    Ec=sent2id(Ec,self.char2id)
                    Ew=sent2id(Ew,self.word2id)

                    As=seq_padding(As,0)
                    Ae=seq_padding(Ae,0)
                    yield Qc,Qw,Ec,Ew,As,Ae
                    Qc,Qw,Ec,Ew,As,Ae=[],[],[],[],[],[]

if __name__=='__main__':
    pass
            




                





                



                
