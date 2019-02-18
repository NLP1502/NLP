import torch as t
import sys
sys.path.append('../')
import csv
import json
import os
import fire
import glob
import torch as t
import numpy as np
from task6_eval_p35 import eval

def getPre(infilePth):
    inValNpz = 'question_context_official_dev_set_enhance3.npz'
    topNum = 10
    inFile = infilePth

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

    fPre = open(inFile+'.pre.csv', 'w')
    flabel = open(inFile + '.label.csv', 'w')
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
        # print(_)
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

def makeTempOutAns(infilename, ansfilename):
    useOpt = 0
    inPreFile = infilename + '.pre.csv'
    inLableFile = infilename + '.label.csv'
    inQuestionFile = './zhihuData/official/dev_data.csv'
    outPreFile = 'TempOutAnsAnalysis.csv'
    outAnsFile = ansfilename
    useOpt = 0
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
            myans = iqfd[i + 1].strip().split(',')[0]
            for _ in range(5):
                myans += ',' + il[_]
            myans += '\n'
            oaf.write(myans)
        if useOpt == 1:
            myans = iqfd[i + 1].strip().split(',')[0]
            for _ in range(3):
                myans += ',' + il[_]
            myans += ',-1'
            myans += ',-1'
            myans += '\n'
            oaf.write(myans)
        if useOpt == 2:
            myans = iqfd[i + 1].strip().split(',')[0]
            for _ in range(4):
                myans += ',' + il[_]
            myans += ',-1'
            myans += '\n'
            oaf.write(myans)
        if useOpt == 3:
            myans = iqfd[i + 1].strip().split(',')[0]
            linesum = 0
            for j in range(5):
                linesum += ipfd[i][j]
            print(linesum)
            if linesum > 5:
                for _ in range(5):
                    myans += ',' + il[_]
                myans += '\n'
                oaf.write(myans)
            elif linesum > 4:
                for _ in range(4):
                    myans += ',' + il[_]
                myans += ',-1'
                myans += '\n'
                oaf.write(myans)
            elif linesum < 0.5:
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

def main(**kwargs):
    maxTop = 1
    weight = 1.015
    devFile = 'zhihuData/official/dev_data.csv'
    outfile = 'score.greedy_officaldev.csv'
    inPath = 'greedy_dev/pth'
    modle_kind_name = 'forOfficalDev'
    print('AAAA')
    for k, v in kwargs.items():
        if k == 'inPath':
            inPath = v
        if k == 'outfile':
            outfile = v

    files = glob.glob(inPath + '/*.pth')
    single_score = {}
    for file in files:
        print(file)
        getPre(file)
        makeTempOutAns(file, 'TempOutAnsOpt0.csv')
        single_score[file] = eval(devFile, 'TempOutAnsOpt0.csv')

    single_score_of_files_rank = []
    for file in files:
        single_score_of_files_rank.append(single_score[file])

    print(single_score_of_files_rank)

    csv_line = []
    for file in files:
        csv_line.append(file)
    f = open(outfile, 'a+', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(csv_line)
    f.close()

    scoreDic = {}
    tNum = len(files)
    csv_line = []
    for _ in range(len(files)):
        this_score = single_score[files[_]]
        scoreDic[str(_)] = this_score
        csv_line.append(this_score)
    f = open(outfile, 'a+', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(csv_line)
    f.close()

    turn = 1
    maxTurn = tNum
    scoreDic_new = scoreDic
    while turn < maxTurn:
        if len(scoreDic_new) == 0:
            break
        scoreDic = scoreDic_new
        scoreDic_new = {}
        rankscorelist = sorted(scoreDic.items(), key=lambda d: d[1], reverse=True)
        maxscore = float(rankscorelist[0][1])
        print(rankscorelist)
        top = 0
        while top < maxTop and top < len(rankscorelist):

            donelist = [int(_) for _ in rankscorelist[top][0].split('.')]
            print(donelist)
            selfscore = rankscorelist[top][1]
            for i in range(tNum):
                if i not in donelist:
                    print(i)
                    if turn == 1:
                        r = weight * t.load(files[donelist[0]])
                    else:
                        r = weight * t.load(os.path.join(inPath, modle_kind_name + '.ensembel' + '_' + rankscorelist[top][0]))
                    r += t.load(files[i])

                    t.save(r, os.path.join(inPath, modle_kind_name + '.tmp'))
                    getPre(os.path.join(inPath, modle_kind_name + '.tmp'))
                    makeTempOutAns(os.path.join(inPath, modle_kind_name + '.tmp'), 'TempOutAnsOpt0.csv')
                    score = eval(devFile, 'TempOutAnsOpt0.csv')


                    print(score)
                    f = open(outfile, 'a+', encoding='utf-8')
                    writer = csv.writer(f)
                    writer.writerow([rankscorelist[top][0], str(i), str(score)])
                    f.close()

                    if score > maxscore:
                        maxscore = score
                        t.save(r, os.path.join(inPath, modle_kind_name + '.ensembel' + '_' + rankscorelist[top][0] + '.' + str(i)))
                        scoreDic_new[rankscorelist[top][0] + '.' + str(i)] = score

            top += 1
        turn += 1

if __name__ == '__main__':
    fire.Fire()


