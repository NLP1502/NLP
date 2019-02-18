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
        if k == 'inQuestionFile':
            inQuestionFile = v
        if k == 'outPreFile':
            outPreFile = v
        if k == 'outAnsFile':
            outAnsFile = v
        if k == 'useOpt':
            useOpt = v
    ipf = open(inPreFile, 'r', encoding='utf8')
    ipf = open(inPreFile, 'r', encoding='utf8')
    ilf = open(inLableFile, 'r', encoding='utf8')
    iqf = open(inQuestionFile, 'r', encoding='utf8')
    opf = open(outPreFile, 'w', encoding='utf8')
    oaf = open(outAnsFile, 'w', encoding='utf8')

    ipfd = ipf.readlines()
    ilfd = ilf.readlines()
    iqfd = iqf.readlines()


    for i in range(len(ipfd)):
        # print(ipfd[i])
        # print(ilfd[i].strip().split(','))
        # print(itad[i+1].strip().split(',')[-1].split('|'))
        il = ilfd[i].strip().split(',')
        # ta = itad[i+1].strip().split(',')[-1].split('|')
        # opf.write(ipfd[i].strip() + ',' + str(p) + '\n')
        if useOpt == 0:
            myans = iqfd[i+1].strip().split(',')[0]
            for _ in range(5):
                myans += ',' + il[_]
            myans += '\n'
            oaf.write(myans)
        if useOpt == 1:
            myans = iqfd[i+1].strip().split(',')[0]
            for _ in range(3):
                myans += ',' + il[_]
            myans += ',-1'
            myans += ',-1'
            myans += '\n'
            oaf.write(myans)
        if useOpt == 2:
            myans = iqfd[i+1].strip().split(',')[0]
            for _ in range(4):
                myans += ',' + il[_]
            myans += ',-1'
            myans += '\n'
            oaf.write(myans)
        if useOpt == 3:
            myans = iqfd[i+1].strip().split(',')[0]
            for _ in range(2):
                myans += ',' + il[_]
            myans += ',-1'
            myans += ',-1'
            myans += ',-1'
            myans += '\n'
            oaf.write(myans)
        if useOpt == 10:
            ansP = ipfd[i].strip().split(',')
            myans = iqfd[i+1].strip().split(',')[0]

            linesum = 0
            for j in range(5):
                linesum += float(ansP[j])
            print(linesum)
            if linesum > 13.5:
                for _ in range(5):
                    myans += ',' + il[_]
                myans += '\n'
                oaf.write(myans)
            elif linesum > 10:
                for _ in range(4):
                    myans += ',' + il[_]
                myans += ',-1'
                myans += '\n'
                oaf.write(myans)
            elif linesum < 6.6:
                for _ in range(2):
                    myans += ',' + il[_]
                myans += ',-1'
                myans += ',-1'
                myans += ',-1'
                myans += '\n'
                oaf.write(myans)
            elif linesum < 3:
                for _ in range(1):
                    myans += ',' + il[_]
                myans += ',-1'
                myans += ',-1'
                myans += ',-1'
                myans += ',-1'
                myans += '\n'
                oaf.write(myans)
            else:
                for _ in range(3):
                    myans += ',' + il[_]
                myans += ',-1'
                myans += ',-1'
                myans += '\n'
                oaf.write(myans)
        if useOpt == 11:
            ansP = ipfd[i].strip().split(',')
            myans = iqfd[i+1].strip().split(',')[0]

            linesum = 0
            for j in range(5):
                linesum += float(ansP[j])
            # print(linesum)
            if linesum > 11.95:
                for _ in range(5):
                    myans += ',' + il[_]
                myans += '\n'
                oaf.write(myans)
            elif linesum > 9.2:
                for _ in range(4):
                    myans += ',' + il[_]
                myans += ',-1'
                myans += '\n'
                oaf.write(myans)
            elif linesum < 6:
                for _ in range(1):
                    myans += ',' + il[_]
                myans += ',-1'
                myans += ',-1'
                myans += ',-1'
                myans += ',-1'
                myans += '\n'
                oaf.write(myans)
            elif linesum < 8:
                for _ in range(2):
                    myans += ',' + il[_]
                myans += ',-1'
                myans += ',-1'
                myans += ',-1'
                myans += '\n'
                oaf.write(myans)
            else:
                for _ in range(3):
                    myans += ',' + il[_]
                myans += ',-1'
                myans += ',-1'
                myans += '\n'
                oaf.write(myans)





if __name__ == '__main__':
    fire.Fire()
