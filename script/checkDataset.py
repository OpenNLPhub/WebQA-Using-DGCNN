import os
import json

cwd=os.getcwd()

def cal_file(filepath:str):
    with open(filepath,'r') as f:
        d=json.loads(f.read())
    print(len(d))

def check_nums():
    root=os.path.join(cwd,'data','WebQA.v1.0')
    train_file_path=os.path.join(root,'me_train.json')
    train_answer_file_path=os.path.join(root,'answer_train.json')
    train_no_answer_file_path=os.path.join(root,'no_answer_train.json')
    test_ann_file_path=os.path.join(root,'me_test.ann.json')
    test_ir_file_path=os.path.join(root,'me_test.ir.json')
    dev_ann_file_path=os.path.join(root,'me_validation.ann.json')
    dev_ir_file_path=os.path.join(root,'me_validation.ir.json')
    cal_file(train_file_path)
    cal_file(test_ann_file_path)
    cal_file(test_ir_file_path)
    cal_file(dev_ann_file_path)
    cal_file(dev_ir_file_path)
    cal_file(train_no_answer_file_path)
    cal_file(train_answer_file_path)


def check_no_answer_in_train_data():
    root=os.path.join(cwd,'data','WebQA.v1.0')
    train_file_path=os.path.join(root,'me_train.json')
    with open(train_file_path,'r') as f:
        d=json.loads(f.read())
    d=list(d.values())

    cnt=0
    for item in d:
        p=item['evidences']
        p=list(p.values())
        f=True
        for ip in p:
            e,a=ip['evidence'],ip['answer']
            if 'no_answer' not in a:
                 cnt+=1
                 break

    print("have answer in evidences {}/{}  {}".format(cnt,len(d),cnt/len(d)))



def pre_process():
    root=os.path.join(cwd,'data','WebQA.v1.0')
    train_file_path=os.path.join(root,'me_train.json')

    with open(train_file_path,'r') as f:
        d=json.loads(f.read())
    d=list(d.values())

    no_answer=[] 
    answer=[]
    for item in d:
        q=item['question']
        p=item['evidences']
        p=list(p.values())
        for ip in p:
            e,a=ip['evidence'],ip['answer']
            i={'question':q,'evidence':e,'answer':a}
            if 'no_answer' in a:
                no_answer.append(i)
            else:
                answer.append(i)
    
    train_answer_file_path=os.path.join(root,'answer_train.json')
    train_no_answer_file_path=os.path.join(root,'no_answer_train.json')
    json.dump(answer,open(train_answer_file_path,'w'))
    json.dump(no_answer,open(train_no_answer_file_path,'w'))

'''

def pre_process_test_dev():
    root=os.path.join(cwd,'data','WebQA.v1.0')
    test_ann_file_path=os.path.join(root,'me_test.ann.json')
    test_ir_file_path=os.path.join(root,'me_test.ir.json')
    dev_ann_file_path=os.path.join(root,'me_validation.ann.json')
    dev_ir_file_path=os.path.join(root,'me_validation.ir.json')


def process(file1,file2):
    with open(file1,'r') as f1,open(file2,'r') as f2:
        F=lambda x: list(x.values())
        d1=F(json.loads(f1.read()))
        d2=F(json.loads(f2.read()))
    d=d1.extend(d2)

'''

def process_ann_test_dev():
    root=os.path.join(cwd,'data','WebQA.v1.0')
    test_ann_file_path=os.path.join(root,'me_test.ann.json')
    dev_ann_file_path=os.path.join(root,'me_validation.ann.json')

    new_test_file_path=os.path.join(root,'answer_test.json')
    new_dev_file_path=os.path.join(root,'answer_dev.json')

    process(test_ann_file_path,new_test_file_path)
    process(dev_ann_file_path,new_dev_file_path)
    
def process(file,outfile):
    with open(file,'r') as f:
        d=list(json.loads(f.read()).values())
    ans=[]
    for item in d:
        q=item['question']
        p=item['evidences']
        p=list(p.values())
        for ip in p:
            e,a=ip['evidence'],ip['answer']
            i={'question':q,'evidence':e,'answer':a}
            ans.append(i)
    json.dump(ans,open(outfile,'w'))
    

    
        
        
if __name__=='__main__':
    # check_nums()
    # check_no_answer_in_train_data()
    pre_process()
    process_ann_test_dev()


