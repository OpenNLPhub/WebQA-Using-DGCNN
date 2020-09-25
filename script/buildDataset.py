import os
import json
import random
from tqdm import tqdm
root = os.path.join(os.getcwd(),'data','WebQA.v1.0')
oroot = os.path.join(os.getcwd(), 'data','dataset')

train = os.path.join(root,'me_train.json')
out_train = os.path.join(oroot,'train.json')

dev = os.path.join(root,'me_validation.ir.json')
out_dev = os.path.join(oroot,'dev.json')

test = os.path.join(root,'me_test.ir.json')
out_test = os.path.join(oroot,'test.json')

out_test_text = os.path.join(oroot, 'test_text.json')

def build_data(p,o):
    with open(p,'r') as f:
        l = json.loads(f.read())
    no_answer_list = []
    answer_list = []
    for k,v in tqdm(l.items()):
        q = v['question']
        es = v['evidences']
        for kk,vv in es.items():
            ee = vv['evidence']
            aa = vv['answer']
            d={}
            d['question'] = q
            d['evidence'] = ee
            d['answer'] = aa

            if "no_answer" in aa:
                no_answer_list.append(d)
            else:
                answer_list.append(d)
    
    # min_len =  len(no_answer_list) if len(no_answer_list) < len(answer_list) else len(answer_list)
    # an = [*no_answer_list[:min_len] , *answer_list[:min_len]]
    an = [*no_answer_list,*answer_list]

    with open(o,'w') as f:
        json.dump(an,f)

def build_test_text(p,o):
    with open(p, 'r') as f:
        l = json.loads(f.read())
    an=[]
    for k,v in tqdm(l.items()):
        d={ }
        q = v['question']
        es = v['evidences']
        e = []
        a = []
        for kk,vv in es.items():
            e.append(vv['evidence'])
            a.extend(vv['answer'])
        a = list(set(a))
        if 'no_answer' in a and len(a)!=0:
            a.remove('no_answer')
        d['question'] = q
        d['evidences'] = e
        d['answer'] = a
        an.append(d)
    with open(o,'w') as f:
        json.dump(an,f)
    
    
if __name__ == '__main__':
    # build_data(train, out_train)
    build_data(test, out_test)
    build_data(dev, out_dev)
    build_test_text(dev, out_test_text)
