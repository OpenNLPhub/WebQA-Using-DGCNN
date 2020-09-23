'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-18 14:06:24
 * @desc 
'''
import os
import numpy as np
import gensim
from tqdm import tqdm
import pickle
cwd=os.getcwd()

def generateWV():
    
    wv_baidu_path=os.path.join(cwd,'data','ChineseWordVec_baike','word2vec_baike')

    model=gensim.models.Word2Vec.load(wv_baidu_path)
    wv=model.wv
    vocab=wv.vocab

    word_path=os.path.join(cwd,'data','ChineseWordVec_baike','word2id.pkl')
    vocab_path=os.path.join(cwd,'data','ChineseWordVec_baike','vocab.txt')
    cnt=2
    word2id={'[PAD]':0,'[UNK]':1}
    with open(vocab_path,'w',encoding='utf-8') as f:
        for w in tqdm(vocab):
            f.write(w+'\n')
            word2id[w]=cnt
            cnt+=1
    
    with open(word_path,'wb') as f:
        pickle.dump(word2id,f)
    
    # import pdb;pdb.set_trace()
    vecs=np.array(wv.vectors)
    #前面需要加入填充词的词向量，我们对其进行随机初始化
    padding=np.zeros((2,vecs.shape[1]))

    embedding_param=np.concatenate([padding,vecs],axis=0)

    embedding_path=os.path.join(cwd,'data','ChineseWordVec_baike','word_embedding.npy')
    np.save(embedding_path,embedding_param)
    print(embedding_param.shape)
    print(cnt)

def generateChar():
    wv_baidu_path=os.path.join(cwd,'data','ChineseWordVec_baike','word2vec_baike')

    model=gensim.models.Word2Vec.load(wv_baidu_path)
    wv=model.wv
    vocab=wv.vocab

    char_path=os.path.join(cwd,'data','ChineseWordVec_baike','char2id.pkl')
    char_emb=[]
    cnt=2
    char2id={'[PAD]':0,'[UNK]':1}

    for w in tqdm(vocab):
        if len(w)==1:
            char_emb.append(wv[w])
            char2id[w]=cnt
            cnt+= 1
    
    with open(char_path,'wb') as f:
        pickle.dump(char2id,f)
    
    # import pdb;pdb.set_trace()
    padding=np.random.randn(2,256)
    char_emb=np.concatenate([padding,char_emb],axis=0)
    embedding_path=os.path.join(cwd,'data','ChineseWordVec_baike','char_embedding.npy')
    np.save(embedding_path,char_emb)
    print(char_emb.shape)
    print(cnt)



if __name__=='__main__':
    generateWV()
    generateChar()
    