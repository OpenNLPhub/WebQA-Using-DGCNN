
from model import DGCNN
import config
from tokenizer import sent2id,tokenize
from utils import alignWord2Char,focal_loss,binary_confusion_matrix_evaluate
import torch
import numpy as np
import torch.optim as optim
from log import logger
from copy import deepcopy
from tabulate import tabulate


class Model(object):
    def __init__(self):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.model = DGCNN(256,char_file = config.char_embedding_path,\
            word_file = config.word_embedding_path).to(self.device)
        self.epoches = 150
        self.lr = 1e-4

        self.print_step = 15
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),\
            lr=self.lr)

        self.best_model = DGCNN(256,char_file=config.char_embedding_path,\
            word_file = config.word_embedding_path).to(self.device)
        self._val_loss = 1e12

        #Debug


    def train(self,train_data,dev_data,threshold=0.1):
        for epoch in range(self.epoches):
            self.model.train()
            for i,item in enumerate(train_data):
                self.optimizer.zero_grad()
                Qc,Qw,q_mask,Ec,Ew,e_mask,As,Ae = [i.to(self.device) for i in item]
                As_, Ae_ = self.model([Qc,Qw,q_mask,Ec,Ew,e_mask])
                As_loss=focal_loss(As,As_,self.device)
                Ae_loss=focal_loss(Ae,Ae_,self.device)
                # batch_size, max_seq_len_e 
                
                mask=e_mask==1
                loss=(As_loss.masked_select(mask).sum()+Ae_loss.masked_select(mask).sum()) / e_mask.sum()
                loss.backward()
                self.optimizer.step()

                if (i+1)%self.print_step==0 or i==len(train_data)-1:
                    logger.info("In Training : Epoch : {} \t Step / All Step : {} / {} \t Loss of every char : {}"\
                        .format(epoch+1, i+1,len(train_data),loss.item()*100))

                #debug
                # if i==2000:
                #     break
            
            self.model.eval()
            with torch.no_grad():
                self.validate(dev_data)
            
    def test(self,test_data,threshold=0.1):
        self.best_model.eval()
        self.best_model.to(self.device)
        with torch.no_grad():
            sl,el,sl_,el_=[],[],[],[]
            for i, item in enumerate(test_data):
                Qc,Qw,q_mask,Ec,Ew,e_mask,As,Ae = [i.to(self.device) for i in item]
                mask=e_mask==1
                As_,Ae_ = self.model([Qc,Qw,q_mask,Ec,Ew,e_mask])
                As_,Ae_,As,Ae = [ i.masked_select(mask).cpu().numpy() for i in [As_,Ae_,As,Ae]]
                As_,Ae_ = np.where(As_>threshold,1,0), np.where(Ae_>threshold,1,0)
                As,Ae = As.astype(int),Ae.astype(int)
                sl.append(As)
                el.append(Ae)
                sl_.append(As_)
                el.append(el_)
            a=binary_confusion_matrix_evaluate(np.concatenate(sl),np.concatenate(sl_))
            b=binary_confusion_matrix_evaluate(np.concatenate(el),np.concatenate(el_))
            logger.info('In Test DataSet: START EVALUATION:\t Acc : {}\t Prec : {}\t Recall : {}\t F1-score : {}'\
                .format(a[0],a[1],a[2],a[3]))
            logger.info('In Test DataSet: START EVALUATION:\t Acc : {}\t Prec : {}\t Recall : {}\t F1-score : {}'\
                .format(b[0],b[1],b[2],b[3]))
                
    def validate(self,dev_data,threshold=0.1):
        val_loss=[]
        # import pdb; pdb.set_trace()
        for i, item in enumerate(dev_data):
            Qc,Qw,q_mask,Ec,Ew,e_mask,As,Ae = [i.to(self.device) for i in item]
            As_, Ae_ =  self.model([Qc,Qw,q_mask,Ec,Ew,e_mask])

            #cal loss
            As_loss,Ae_loss=focal_loss(As,As_,self.device) ,focal_loss(Ae,Ae_,self.device)
            mask=e_mask==1
            loss=(As_loss.masked_select(mask).sum() + Ae_loss.masked_select(mask).sum()) /  e_mask.sum()
            if (i+1)%self.print_step==0 or i==len(dev_data)-1:
                logger.info("In Validation: Step / All Step : {} / {} \t Loss of every char : {}"\
                    .format(i+1,len(dev_data),loss.item()*100))
            val_loss.append(loss.item())
            
            
            As_,Ae_,As,Ae = [ i.masked_select(mask).cpu().numpy() for i in [As_,Ae_,As,Ae]]
            As_,Ae_ = np.where(As_>threshold,1,0), np.where(Ae_>threshold,1,0)
            As,Ae = As.astype(int),Ae.astype(int)
            
            acc,prec,recall,f1=binary_confusion_matrix_evaluate(As,As_)
            
            logger.info('START EVALUATION :\t Acc : {}\t Prec : {}\t Recall : {}\t F1-score : {}'\
                .format(acc,prec,recall,f1))
            acc,prec,recall,f1=binary_confusion_matrix_evaluate(Ae,Ae_)
            logger.info('END EVALUATION :\t Acc : {}\t Prec : {}\t Recall : {}\t F1-score : {}'\
                .format(acc,prec,recall,f1))
            # [ , seq_len]
        l=sum(val_loss)/len(val_loss)
        logger.info('In Validation, Average Loss : {}'.format(l*100))
        if l<self._val_loss:
            logger.info('Update best Model in Valiation Dataset')
            self._val_loss=l
            self.best_model=deepcopy(self.model)


    def load_model(self,PATH):
        self.best_model.load_state_dict(torch.load(PATH))
        self.best_model.eval()

    def save_model(self,PATH):
        torch.save(self.best_model.state_dict(),PATH)
        logger.info('save best model successfully')

    '''
    这里的Data是指含有原始文本的数据List[ dict ]
    - test_data
    | - { 'question', 'evidences', 'answer'}
    '''
    def get_test_answer(self,test_data,word2id,char2id):
        all_item =  len(test_data)
        t1=0.
        t3=0.
        t5=0.
        self.best_model.eval()
        with torch.no_grad():
            for item in test_data:
                q_text = item['question']
                e_texts = item['evidences']
                a = item['answer']
                a_ = extract_answer(q_text,e_texts,word2id,char2id)
                # a_  list of [ answer , possibility]
                n=len(a_)

                a_1 = {i[0] for i in a_[:1]}
                a_3 = {i[0] for i in a_[:3]}
                a_5 = {i[0] for i in a_[:5]}

                if a[0] == 'no_answer' and n==0:
                    t1+=1
                    t3+=1
                    t5+=1
                
                if [i for i in a if i in a_1]:
                    t1+=1
                if [i for i in a if i in a_3]:
                    t3+=1
                if [i for i in a if i in a_5]:
                    t5+=1
        
        logger.info('In Test Raw File')
        logger.info('Top One Answer : Acc : {}'.format(t1/all_item))
        logger.info('Top Three Answer : Acc : {}'.format(t3/all_item))
        logger.info('Top Five Answer : Acc : {}'.format(t5/all_item))
        
    def extract_answer(self,q_text,e_texts,word2id,char2id,maxlen=10,threshold=0.1):
        Qc,Qw,Ec,Ew= [],[],[],[]
        qc = list(q_text)
        Qc,q_mask=sent2id([qc],char2id)

        qw = alignWord2Char(tokenize(q_text))
        Qw,q_mask_=sent2id([qw],word2id)

        assert torch.all(q_mask == q_mask_)

        tmp = [(list(e),alignWord2Char(tokenize(e))) for e in e_texts]
        ec,ew = zip(*tmp)

        Ec,e_mask=sent2id(list(ec),char2id)
        Ew,e_mask_=sent2id(list(ew),word2id)
        assert torch.all(e_mask == e_mask_)

        totensor=lambda x: torch.from_numpy(np.array(x)).long()

        L=[Qc,Qw,q_mask,Ec,Ew,e_mask]
        L=[totensor(x) for x in L]

        As_ , Ae_ = self.best_model(L)

        R={}
        for as_ ,ae_ , e in zip(As_,Ae_,e_texts):
            as_ ,ae_ = as_[:len(e)].numpy() , ae_[:len(e)].numpy()
            sidx = torch.where(as_>threshold)[0]
            eidx = torch.where(ae_>threshold)[0]
            result = { }
            for i in sidx:
                cond = (eidx >= i) & (eidx < i+maxlen)
                for j in eidx[cond]:
                    key=e[i:j+1]
                    result[key]=max(result.get(key,0),as_[i] * ae_[j])
            if result:
                for k,v in result.items():
                    if k not in R:
                        R[k]=[]
                    R[k].append(v)        
        # sort all answer
        R= [
            [k,((np.array(v)**2).sum()/(sum(v)+1))]
            for k , v in R.items()
        ]

        R.sort(key=lambda x: x[1], reversed=True)
        # R 降序排列的 (answer, possibility)
        return R





if __name__ == '__main__':
    pass



            


        