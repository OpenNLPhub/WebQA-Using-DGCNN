import os
import json

cwd=os.getcwd()

def cal_file(filepath:str):
    with open(filepath,'r') as f:
        d=json.loads(f.read())
    print(len(d))

if __name__=='__main__':
    root=os.path.join(cwd,'data','WebQA.v1.0')
    train_file_path=os.path.join(root,'me_train.json')
    test_ann_file_path=os.path.join(root,'me_test.ann.json')
    test_ir_file_path=os.path.join(root,'me_test.ir.json')
    dev_ann_file_path=os.path.join(root,'me_validation.ann.json')
    dev_ir_file_path=os.path.join(root,'me_validation.ir.json')

    cal_file(train_file_path)
    cal_file(test_ann_file_path)
    cal_file(test_ir_file_path)
    cal_file(dev_ann_file_path)
    cal_file(dev_ir_file_path)
    
    pass

