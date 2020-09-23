
import config
import numpy as np
from tokenizer import seq_padding,sent2id,tokenize
import random
import re
import torch
from utils import alignWord2Char
import json



class data_generator(object):
    def __init__(self,data_file,word2id,char2id,batch_size=config.batch_size):
        with open(data_file,'r') as f:
            data=json.loads(f.read())
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
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        Qc,Qw,Ec,Ew,As,Ae=[],[],[],[],[],[]
        for i in idxs:
            item=self.data[i]

            #question
            q=item['question']
            q_char=list(q)
            #重复单词个数，使q_char 和 q_word 保持相同的长度
            q_word=tokenize(q)
            q_word=alignWord2Char(q_word)
            assert len(q_char)==len(q_word)
            
            #evidence
            e=item['evidence']
            e_char=list(e)
            e_word=tokenize(e)
            e_word=alignWord2Char(e_word)
            assert len(e_char)==len(e_word)
            
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
                # import pdb;pdb.set_trace()
                Qc,q_mask=sent2id(Qc,self.char2id)
                Qw,q_mask_=sent2id(Qw,self.word2id)
                assert np.all(q_mask==q_mask_)

                Ec,e_mask=sent2id(Ec,self.char2id)
                Ew,e_mask_=sent2id(Ew,self.word2id)
                assert np.all(e_mask==e_mask_)

                # import pdb;pdb.set_trace()
                As,_=seq_padding(As,0)
                Ae,_=seq_padding(Ae,0)
                # answer padding == evidence padding

                totensor=lambda x: torch.from_numpy(np.array(x))
                q_mask=totensor(q_mask).long()
                e_mask=totensor(e_mask).long()
                Qc=totensor(Qc).long()
                Qw=totensor(Qw).long()
                Ec=totensor(Ec).long()
                Ew=totensor(Ew).long()
                As=totensor(As).float()
                Ae=totensor(As).float()

                yield [Qc,Qw,q_mask,Ec,Ew,e_mask,As,Ae]
                Qc,Qw,Ec,Ew,As,Ae=[],[],[],[],[],[]

if __name__=='__main__':
    #debug Test
    from tokenizer import get_Map_char_id,get_Map_word_id
    _, word2id, _ = get_Map_word_id()
    _, char2id, _ = get_Map_char_id()
    train_data=data_generator(config.train_path,word2id,char2id)
    import pdb;pdb.set_trace()
    for idx,item in enumerate(train_data):
        Qc,Qw,q_mask,Ec,Ew,e_mask,As,Ae = item

            




                
 




                



                
