import os



cwd=os.getcwd()


'''----------------------- Path Config ----------------------------'''

char_path=os.path.join(cwd,'data','ChineseWordVec_baike','char.txt')
vocab_path=os.path.join(cwd,'data','ChineseWordVec_baike','vocab.txt')
word_embedding_path=os.path.join(cwd,"data","ChineseWordVec_baike",'word_embedding.npy')
char_embedding_path=os.path.join(cwd,"data","ChineseWordVec_baike",'char_embedding.npy')

data_root=os.path.join(cwd,'data','WebQA.v1.0')
train_data_path=os.path.join(data_root,'me_train.json')
test_data_ann_path=os.path.join(data_root,'me_test.ann.json')
test_data_ir_path=os.path.join(data_root,'me_test.ir.json')
dev_data_ann_path=os.path.join(data_root,'me_validation.ann.json')
dev_data_ir_path=os.path.join(data_root,'me_validation.ir.json')



'''----------------------- Training Config ----------------------------'''

batch_size=32
max_seq_len=256