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
cwd=os.getcwd()

def generateWV():
    
    wv_baidu_path=os.path.join(cwd,'data','ChineseWordVec_baike','word2vec_baike')

    model=gensim.models.Word2Vec.load(wv_baidu_path)
    wv=model.wv
    vocab=wv.vocab

    vocab_path=os.path.join(cwd,'data','ChineseWordVec_baike','vocab.txt')

    with open(vocab_path,'w') as f:
        #前面写入填充词 0个为PAD 因为在Embbeding 层，我们将padding_idx=0
        f.write("[PAD]\n")
        f.write("[UNK]\n")
        for w in tqdm(vocab):
            f.write(w+'\n')
    

    vecs=np.array(wv.vectors)
    #前面需要加入填充词的词向量，我们对其进行随机初始化
    padding=np.zeros(2,wv.vectors.shape[1])

    embedding_param=np.concatenate([padding,vecs],axis=0)

    embedding_path=os.path.join(cwd,'data','ChineseWordVec_baike','word_embedding.npy')
    np.save(embedding_path,embedding_param)

# def generateChar():
#     wv_baidu_path=os.path.join(cwd,'data','ChineseWordVec_baike','word2vec_baike')

#     model=gensim.models.Word2Vec.load(wv_baidu_path)
#     wv=model.wv
#     vocab=wv.vocab

#     char_path=os.path.join(cwd,'data','ChineseWordVec_baike','char.txt')
#     char_emb=[]
#     with open(char_path,'w') as f:
#         #前面写入填充词 0个为PAD 因为在Embbeding 层，我们将padding_idx=0
#         f.write("[PAD]\n")
#         f.write("[UNK]\n")
#         for w in tqdm(vocab):
#             if len(w)==1:
#                 char_emb.append(wv[w])
#                 f.write(w+'\n')

#     padding=np.random.randn(2,wv.vectors.shape[1])
#     char_emb=np.concatenate([padding,char_emb],axis=0)
#     embedding_path=os.path.join(cwd,'data','ChineseWordVec_baike','char_embedding.npy')
#     np.save(embedding_path,char_emb)

if __name__=='__main__':
    generateWV()
    # generateChar()
    