# WebQA-Using-DGCNN
Reimplementation of DGCNN using PyTorch.
The idea come from [Jianlin. Su 's Blog](https://spaces.ac.cn/archives/5409)
DGCNN is based on CNN and simple Attention mechanism.
It is very efficient and lightweight, because of no RNN architecture in this model.
It is designed for WebQA Task specificly.

### Dataset
I use dataset is WebQA BaiDu Reaserch's Paper 
- Peng Li, Wei Li, Zhengyan He, Xuguang Wang, Ying Cao, Jie Zhou, and Wei Xu. 2016. Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering.[arXiv:1607.06275](https://arxiv.org/abs/1607.06275)

Thanks for their sharing

Based on above dataset, Jianlin. Su. create a pure version which maybe more suitable for student.
He processed these raw data and shared them on his [blog](https://kexue.fm/archives/4338)

Bacause the dataset is too large, I did not push it to github
You can download the Dataset according to Su's blog. And then move these data under data directory.

The Structure of WebQA
\- WebQA
 |\- readme.md
 |\- me_test.ann.json
 |\- me_test.ir.json
 |\- me_train.json
 |\- me_validation.ann.json
 |\- me_validation.ir.json
The difference between ir and ann is that:
In ann dataset every data item have one question and coressponding one evidence which have answer
In ir dataset every data item have one question and multiple evidences which may not have true ansewer

The stucture of json file:
- Using json.loads read file, get a dict. The key of this dict for exmaple "Q_TRN_010878" is the index of question
- We can get data item through d['Q_TRN_010878']. Every data item is a dict which has two keys : 'question' and 'evidences'
- Through d['Q_TRN_010878']['question'], we can get the text of question. For exmaple: "勇敢的心霍笑林的父亲是谁出演的"
- Through d['Q_TRN_010878']['evidences'], we can get a dict. The key of this dict is the index of evidence
- Through d['Q_TRN_010878']['evidences']['Q_TRN_010878#05'],we can also get a dict, which has two keys: 'evidence' and 'answer'
- evidece is the text of evidence, answer is a list (maybe not only one answer). If no answer, answer is ['no_answer']
  
|  filename |  itemNums |
| ---- | ---- |
|me_train.json|36181|
|me_validation.json|3018|
|me_test.json|3024|

