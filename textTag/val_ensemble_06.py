import torch as t
import sys
sys.path.append('../')
from utils import get_score
import glob
import torch as t
import numpy as np
import json
import time
label_path =   'labels_06.json'
test_data_path='test_06.npz'
index2qid = np.load(test_data_path)['index2qid'].item()
with open(label_path) as f:
      labels_info = json.load(f)
qid2label = labels_info['d']
label2qid = labels_info['id2label']
files = glob.glob('./result/*.pth')
r=0
for file in files:
    print(file)
    r+=t.load(file)
    # print(r)

# true_labels = [qid2label[index2qid[650243-200000+ii]] for ii in range(len(r))]
true_labels = []
for ii in range(len(r)):
    true_labels.append(qid2label[index2qid[ii]])

tmp = r
result=(tmp).topk(5,1)[1]
# 计算在验证集上的分数 0.43065  提交后在0.43565以上
predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result,true_labels)]
# print(predict_label_and_marked_label_list)
score,_,_,ss = get_score(predict_label_and_marked_label_list)
print(score)
print(ss)