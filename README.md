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



