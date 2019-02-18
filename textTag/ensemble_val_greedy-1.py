import torch as t
import sys
sys.path.append('../')
from utils import get_score
import csv
import json
import os
from config_06 import opt
import fire
import glob
import torch as t
import numpy as np

def main(**kwargs):
    maxTop = 4
    weight = 1

    opt.parse(kwargs)
    outfile = opt.ensemble_outfile
    label_path = opt.labels_path
    test_data_path= opt.test_data_path
    single_model_score_file = opt.single_model_score_file
    pth_data_path = opt.inpath
    index2qid = np.load(test_data_path)['index2qid'].item()
    with open(label_path) as f:
          labels_info = json.load(f)
    qid2label = labels_info['d']
    label2qid = labels_info['id2label']
    files = glob.glob(pth_data_path +'/*.pth')
    print(files)
    modle_kind_name = 'baseline_greedy'

    single_score = {}
    fin = open(single_model_score_file, 'r', encoding='utf8')
    reader = csv.reader(fin)
    single_model_score_file_data = list(reader)
    for _ in range(len(single_model_score_file_data)):
        single_score[single_model_score_file_data[_][0].split('/')[-1].strip()] = float(single_model_score_file_data[_][1])

    print(single_score)
    csv_line = []
    single_score_of_files_rank = []
    for file in files:
        csv_line.append(os.path.split(file)[1])
        single_score_of_files_rank.append(single_score[os.path.split(file)[1][::-1].split('.',1)[1][::-1]] )
    f = open(outfile, 'a+', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(csv_line)
    f.close()

    scoreDic = {}
    tNum = len(files)
    dNum = 0
    csv_line = []
    for _ in range(len(files)):
        this_score = single_score[files[_].split('/')[-1].strip().replace('.pth','')]
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
                    print(os.path.join(opt.inpath, modle_kind_name + '.ensembel' + '_' + rankscorelist[top][0]))
                    if turn == 1:
                        r = weight * t.load(files[donelist[0]])
                    else:
                        r = weight * t.load(os.path.join(opt.inpath, modle_kind_name + '.ensembel' + '_' + rankscorelist[top][0]))
                    r += t.load(files[i])
                    true_labels = []

                    for ii in range(len(r)):
                        # print(ii)
                        # print(index2qid[ii])
                        # print(qid2label[index2qid[ii]])
                        print(len(r))
                        true_labels.append(qid2label[index2qid[ii]])
                    result = (r).topk(5, 1)[1]
                    predict_label_and_marked_label_list = [[_1, _2] for _1, _2 in zip(result, true_labels)]
                    score, _, _, ss = get_score(predict_label_and_marked_label_list)

                    print(score)
                    f = open(outfile, 'a+', encoding='utf-8')
                    writer = csv.writer(f)
                    writer.writerow([rankscorelist[top][0], str(i), str(score)])
                    f.close()

                    if score > maxscore:
                        maxscore = score
                        t.save(r, os.path.join(opt.inpath, modle_kind_name + '.ensembel' + '_' + rankscorelist[top][0] + '.' + str(i)))
                        scoreDic_new[rankscorelist[top][0] + '.' + str(i)] = score

            top += 1
        turn += 1



if __name__ == '__main__':
    fire.Fire()