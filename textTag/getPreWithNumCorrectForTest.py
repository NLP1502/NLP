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
        if k == 'topNum':
            topNum = v


    index2qid = np.load(inValNpz)['index2qid'].item()
    # qid2label = json.load(open(queTopicJson,'r'))['d']
    foutqidnum = open('qid2num', 'w', encoding='utf8')

    tlenth = len(index2qid)
    index2lenList = []
    index2trueList = []
    # for _ in range(tlenth):
    #     qid = index2qid[_]
    #     true_ans = qid2label[qid]
    #     foutqidnum.write(index2qid[_] + ',' + str(len(true_ans)) + '\n')
    #     index2lenList.append(len(true_ans))
    #     index2trueList.append(true_ans)

    fPre = open(inFile+'.pre.csv', 'w', encoding='utf8')
    flabel = open(inFile + '.label.csv', 'w', encoding='utf8')
    data = t.load(inFile)
    top = (data).topk(topNum, 1)
    # import ipdb;ipdb.set_trace()

    for _ in range(tlenth):
        l = ''
        geti = []
        for u in range(topNum):
            # import ipdb;ipdb.set_trace()
            l += str(top[0][_][u]) + ','
            find = 0
            # import ipdb;ipdb.set_trace()
            # for iii in range(len(index2trueList[_])):
            #     if  top[1][_][u]+1  == index2trueList[_][iii]:
            #         geti.append(iii)
            #         find = 1
            #         print(iii)
            #         break
            # if find == 0:
            #     geti.append(0)
        print(_)
        # l += str(index2lenList[_])
        # for u in range(len(geti)):
        #     l += str(geti[u]) + ','
        l += '\n'

        fPre.write(l)
        l = ''
        for u in range(topNum):
            l += str(top[1][_][u] + 1) + ','
        l += '\n'
        flabel.write(l)
    # print(data[0])
    # print(data[1])
    fPre.close()
    flabel.close()


if __name__ == '__main__':
    fire.Fire()
