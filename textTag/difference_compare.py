#!/usr/bin/env python
#coding=utf8


import csv 
import math
import sys
import pickle

def compare(resultFileA, resultFileB, savepath, labelNumStr):
    labelNum = int(labelNumStr) + 1
    print(savepath)
    frFA = open(resultFileA, 'rb')
    frFB = open(resultFileB, 'rb')
    # {'r': label_right, 'f': label_false}

    rFA = pickle.load(frFA)
    rFB = pickle.load(frFB)
    # import ipdb;ipdb.set_trace()
    rd = 0
    fd = 0
    print(len(rFA['r']))
    sumAr = 0
    sumAf = 0
    sumBr = 0
    sumBf = 0
    for i in range(labelNum):
        sumAr += rFA['r'][i]
        sumAf += rFA['f'][i]
        sumBr += rFB['r'][i]
        sumBf += rFB['f'][i]
    print("sumAr: {}, sumAf: {}, sumBr: {}, sumBf: {}".format(sumAr, sumAf, sumBr, sumBf))
    for i in range(labelNum):
        rd += pow(rFA['r'][i]/sumAr - rFB['r'][i]/sumBr, 2)
    for i in range(labelNum):
        fd += pow(rFA['f'][i]/sumAf - rFB['f'][i]/sumBf, 2)
    print("rd: {}, fd: {}".format(rd, fd))

if __name__=='__main__':
    print(sys.argv)
    compare(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])