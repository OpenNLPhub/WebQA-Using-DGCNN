# WebQA-Using-DGCNN
Reimplementation of DGCNN using PyTorch.
The idea come from [Jianlin. Su 's Blog](https://spaces.ac.cn/archives/5409)
Dilate Gated Convolutional Neural Network (DGCNN) is based on CNN and simple Attention mechanism.
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
|answer_train.json|140897|
|no_answer_train.json|307547|



### Data Preprocess

 This Raw Data is not convinient for us.  For trainining dataset, I seperate the evidences.Every data item has one question, one evidence and an answer list. Some items have answer but some do not. Their ratio is 1:1。

For validation dataset and test dataset, I just seperate the evidences,keep one item having one question,one evidence and an answer list.

Just Like this:

```json
[
  {
    "question":"世界第一高峰是什么?",
    "evidence":"世界上的第一高峰是珠穆朗玛峰",
  	"answer":["珠穆朗玛峰"]
	},
  ...
  {
    "question":"世界第一高峰是什么?",
    "evidencee":"武夷山很高",
    "answer":["no_answer"]
  }
]
```



### Usage

**Download Dataset and Pretrained word vector**

 (WordVector I use BaiDuBaike https://pan.baidu.com/s/1YYE2T3f-lPyLBrJuUowAsA  Password: 5p0h)

Download them and Place them under data directory



**Preprocess the WordVector and Dataset**

do

```shell
python script/buildDataset.py

python script/generateWordVec.py
```

You will find that

Under *data* directory there is an aditional directory dataset

Under *ChinsesWordVec_baike* directory there are four additional file *word_embedding.npy*, *char_embedding.npy*, *char2id.pkl*, *word2id.pkl*



**Training and Prediction**

```shell
python src/run.py
```

You can find log information in you terminal and log file under log directory.

The trained Model is stored under result directoy.



### Problem 

I failed to reimplement the expected result of this model.

i found using pointer-label model , the final classifier trends to predict all item to zero.'

Finally, in validation step, the f1-score, recall, precesion are all zero.

**update**

2020/09/24	 proposing a issue in https://github.com/bojone/dgcnn_for_reading_comprehension/issues/5

2020/09/25     try to adjust some random seed to give a shot 2333.

