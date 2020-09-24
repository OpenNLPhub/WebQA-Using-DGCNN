import os



cwd=os.getcwd()


'''----------------------- Path Config ----------------------------'''
wv_baidu_path=os.path.join(cwd,'data','ChineseWordVec_baike','word2vec_baike')

#pretraind data path
vocab_path = os.path.join(cwd,'data','ChineseWordVec_baike','vocab.txt')
char_path = os.path.join(cwd,'data','ChineseWordVec_baike','char2id.pkl')
word_path = os.path.join(cwd,'data','ChineseWordVec_baike','word2id.pkl')
word_embedding_path = os.path.join(cwd,"data","ChineseWordVec_baike",'word_embedding.npy')
char_embedding_path = os.path.join(cwd,"data","ChineseWordVec_baike",'char_embedding.npy')


# data path
raw_data_root=os.path.join(cwd,'data','WebQA.v1.0')

train_raw_path = os.path.join(raw_data_root,'me_train.json')
test_raw_ann_path = os.path.join(raw_data_root,'me_test.ann.json')
test_raw_ir_path = os.path.join(raw_data_root,'me_test.ir.json')
dev_raw_ann_path = os.path.join(raw_data_root,'me_validation.ann.json')
dev_raw_ir_path = os.path.join(raw_data_root,'me_validation.ir.json')

dataset_root = os.path.join(cwd,'data','dataset')
train_path = os.path.join(dataset_root,'train.json')
dev_path = os.path.join(dataset_root,'dev.json')
test_path = os.path.join(dataset_root,'test.json')
test_text_path =  os.path.join(dataset_root,'test_text.json')


#Model Path
model_path=os.path.join(cwd,'result','dgcnn.pth')

'''----------------------- Training Config ----------------------------'''

batch_size=64
max_seq_len=256

