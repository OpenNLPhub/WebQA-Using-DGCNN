from generate import data_generator
import config as config
from tokenizer import get_Map_char_id,get_Map_word_id
from wrapper import Model
import os
import json

def run():
    _, word2id, _ = get_Map_word_id()
    _, char2id, _ = get_Map_char_id()
    train_data = data_generator(config.train_path, word2id, char2id)
    test_data = data_generator(config.test_path, word2id, char2id)
    dev_data =  data_generator(config.dev_path, word2id, char2id)
    
    M=Model()
    if os.path.exists(config.model_path):
        M.load_model(config.model_path)
    else:
        M.train(train_data,dev_data)
    M.test(test_data)

    with open(config.test_text_path,'r',encoding='utf-8') as f:
        test_text_data=json.loads(f.read())
    M.get_test_answer(test_text_data,word2id,char2id)

if __name__ == '__main__':
    run()
