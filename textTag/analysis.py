import fire
import glob
import os
import torch as t
import numpy as np
import json
import csv
def main(**kwargs):
    useOpt = 0
    for k, v in kwargs.items():
        if k == 'inPreFile':
            inPreFile = v
        if k == 'inLableFile':
            inLableFile = v
        if k == 'inTrueAns':
            inTrueAns = v
        if k == 'outPreFile':
            outPreFile = v
        if k == 'outAnsFile':
            outAnsFile = v
        if k == 'useOpt':
            useOpt = v

    ipf = open(inPreFile, 'r', encoding='utf8')
    ilf = open(inLableFile, 'r', encoding='utf8')
    ita = open(inTrueAns, 'r', encoding='utf8')
    opf = open(outPreFile, 'w', encoding='utf8')
    oaf = open(outAnsFile, 'w', encoding='utf8')

    ipfd = ipf.readlines()
    ilfd = ilf.readlines()
    itad = ita.readlines()

    # import ipdb;ipdb.set_trace()
    for i in range(len(ipfd)):
        # print(ipfd[i])
        # print(ilfd[i].strip().split(','))
        # print(itad[i+1].strip().split(',')[-1].split('|'))
        il = ilfd[i].strip().split(',')
        ta = itad[i+1].strip().split(',')[-1].split('|')
        p = 0
        for _ in range(5):
            if il[_] in ta:
                p = _+1
        opf.write(ipfd[i].strip() + ',' + str(p) + '\n')
        if useOpt == 0:
            myans = itad[i+1].strip().split(',')[0]
            for _ in range(5):
                myans += ',' + il[_]
            myans += '\n'
            oaf.write(myans)
        if useOpt == 1:
            myans = itad[i+1].strip().split(',')[0]
            for _ in range(3):
                myans += ',' + il[_]
            myans += ',-1'
            myans += ',-1'
            myans += '\n'
            oaf.write(myans)





if __name__ == '__main__':
    fire.Fire()
