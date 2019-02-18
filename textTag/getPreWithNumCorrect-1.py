import fire
import glob
import os
import torch as t
import numpy as np
import json
import csv
def main(**kwargs):
    for k, v in kwargs.items():
        if k == 'inFile':
            inFile = v
        if k == 'inValNpz':
            inValNpz = v
        if k == 'queTopicJson':
            queTopicJson = v
        if k == 'topNum':
            topNum = v


    index2qid = np.load(inValNpz)['index2qid'].item()
    qid2label = json.load(open(queTopicJson,'r'))['d']
    foutqidnum = open('qid2num', 'w', encoding='utf8')

    tlenth = len(index2qid)
    index2lenList = []
    for _ in range(tlenth):
        qid = index2qid[_]
        true_ans = qid2label[qid]
        foutqidnum.write(index2qid[_] + ',' + str(len(true_ans)) + '\n')
        index2lenList.append(len(true_ans))

    flist = open(inFile+'.list.csv', 'w')
    data = t.load(inFile)
    top = (data).topk(topNum, 1)
    import ipdb;
    ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    for _ in range(tlenth):
        l = ''
        for u in range(topNum):
            l += str(data[0][_][u]) + ','
        l += str(index2lenList[_]) + '\n'
        flist.write(l)
    # print(data[0])
    # print(data[1])
    flist.close()


if __name__ == '__main__':
    fire.Fire()
