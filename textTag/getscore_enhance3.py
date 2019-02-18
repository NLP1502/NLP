import csv
import fire
import sys
import os
sys.path.append('../')
from utils import get_score
import glob
import torch as t
import numpy as np
import json

def main(**kwargs):
    print("@@@@@@@@@@@@@@@@@@@@@@@")
    inpath = 'result'
    outfile = 'score.csv'
    label_path = 'labels_enhanceA.json'
    test_data_path = 'test_enhanceA.npz'
    for k, v in kwargs.items():
        print(k)
        print(v)
        if k == 'inpath':
            inpath = v
        if k == 'outfile':
            outfile = v
        if k == 'label_path':
            label_path = v
        if k == 'test_data_path':
            test_data_path = v
    print(label_path)
    if inpath[-1] == '/':
        inpath = inpath[0:-1]
    index2qid = np.load(test_data_path)['index2qid'].item()
    with open(label_path) as f:
          labels_info = json.load(f)
    qid2label = labels_info['d']
    label2qid = labels_info['id2label']
    files = glob.glob(inpath + '/*.pth')
    print(files)
    for file in files:
        f = open(outfile, 'a', encoding='utf-8')
        print(file)
        if not os.path.isfile(file):
            print('is path')
            continue
        r = t.load(file)
        true_labels = []
        for ii in range(len(r)):
            true_labels.append(qid2label[index2qid[ii]])
        tmp = r
        result=(tmp).topk(5,1)[1]
        predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result,true_labels)]
        score,_,_,ss = get_score(predict_label_and_marked_label_list)
        print(score)
        print(ss)
        writer = csv.writer(f)
        writer.writerow([file,str(score)])
        f.close()

if __name__ == '__main__':
    print('!!!!!!!!!!')
    fire.Fire()