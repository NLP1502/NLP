import fire
import glob
import os
import torch as t
import numpy as np
import json
import csv
def main(**kwargs):
    for k, v in kwargs.items():
        if k == 'inpath':
            inpath = v
        if k == 'inValNpz':
            inValNpz = v
        if k == 'queTopicJson':
            queTopicJson = v
        if k == 'topNum':
            topNum = v


    files = glob.glob(os.path.join(inpath, '*.topPrId' + str(topNum)))

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



    total = len(files)
    for file in files:
        print(file)
        flist = open(file+'.list.csv', 'w')
        data = t.load(file)
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
