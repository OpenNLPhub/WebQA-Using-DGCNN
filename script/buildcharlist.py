

import os
import json
cwd=os.getcwd()
root=os.path.join(cwd,'data','WebQA.v1.0')

def checkbaikecharlist():
    p=os.path.join(cwd,'data','ChineseWordVec_baike','char.txt')
    with open(p,'r') as f:
        lines=f.readlines()
    print(len(lines))

if __name__=='__main__':
    no_trainPath=os.path.join(root,'no_answer_train.json')
    trainPath=os.path.join(root,'answer_train.json')
    devPath=os.path.join(root,'answer_dev.json')
    testPath=os.path.join(root,'answer_test.json')
    charlistPath=os.path.join(root,'char.txt')
    charlist={}
    for p in [trainPath,devPath,testPath,no_trainPath]:
        with open(p,'r') as f:
            d=json.loads(f.read())
        for item in d:
            q=item['question']
            e=item['evidence']
            for i in list(q+e):
                if i==' ':
                    continue
                if i not in charlist:
                    charlist[i]=1
                    
    print("char list len:{}".format(len(charlist)))
    chars=list(charlist.keys())
    with open(charlistPath,'w') as f:
        for i in chars:
            f.write(i+'\n')


