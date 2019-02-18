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

    single_score = {}
    fin = open(single_model_score_file, 'r', encoding='utf8')
    reader = csv.reader(fin)
    single_model_score_file_data = list(reader)
    for _ in range(len(single_model_score_file_data)):
        single_score[single_model_score_file_data[_][0].split('/')[-1].strip()] = float(single_model_score_file_data[_][1])

    model_kind_list = []
    model_kind_dict = {}
    for i in range(len(files)):
        filename = os.path.split(files[i])[1]
        model_kind = filename[::-1].split('_', 1)[1][::-1].strip()
        if model_kind in model_kind_list:
            model_kind_dict[model_kind].append(files[i])
        else:
            model_kind_list.append(model_kind)
            model_kind_dict[model_kind] = []
            model_kind_dict[model_kind].append(files[i])

    same_modle_ensemble_file_list = []
    same_modle_ensemble_score_dict = {}
    print('model_kind_list: ')
    print(model_kind_list)
    t_m_num = len(model_kind_list)
    d_m_num = 0
    for deal_model_list in model_kind_list:
        print('dealing %s model'%(deal_model_list))
        d_m_num += 1
        r = 0
        csv_line = []
        t_c_num = len(model_kind_dict[deal_model_list])
        d_c_num = 0
        for file in model_kind_dict[deal_model_list]:
            csv_line.append(file.split('/')[-1].strip())
        csv_line.append(deal_model_list + '.ensembel')
        f = open(outfile, 'a+', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(csv_line)
        f.close()
        csv_line = []
        for file in model_kind_dict[deal_model_list]:
            d_c_num += 1
            print('dealing %d/%d model: %d/%d checkpoint'%(d_m_num, t_m_num, d_c_num, t_c_num))
            r += t.load(file)
            this_score = str(single_score[file.split('/')[-1].strip().replace('.pth','')])
            print(this_score)
            csv_line.append(this_score)
        print('counting')
        t.save(r, os.path.join(opt.inpath, deal_model_list + '.ensembel'))
        true_labels = []
        for ii in range(len(r)):
            true_labels.append(qid2label[index2qid[ii]])
        result = (r).topk(5, 1)[1]
        predict_label_and_marked_label_list = [[_1, _2] for _1, _2 in zip(result, true_labels)]
        score, _, _, ss = get_score(predict_label_and_marked_label_list)
        print(score)
        csv_line.append(str(score))
        f = open(outfile, 'a+', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(csv_line)
        f.close()
        same_modle_ensemble_file_list.append(deal_model_list + '.ensembel')
        same_modle_ensemble_score_dict[deal_model_list + '.ensembel'] = score

    csv_line = []
    csv_line.append('')
    for _ in same_modle_ensemble_file_list:
        csv_line.append(_)
    f = open(outfile, 'a+', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(csv_line)
    f.close()

    for i in range(len(same_modle_ensemble_file_list)):
        for j in range(len(same_modle_ensemble_file_list)):
            csv_line = []
            csv_line.append(same_modle_ensemble_file_list[i])
            if i > j:
                print('dealing: %d, %d'% (i, j))
                r = t.load(os.path.join(opt.inpath, same_modle_ensemble_file_list[i]))
                r += t.load(os.path.join(opt.inpath, same_modle_ensemble_file_list[j]))
                true_labels = []
                for ii in range(len(r)):
                    true_labels.append(qid2label[index2qid[ii]])
                result=(r).topk(5,1)[1]
                predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result,true_labels)]
                score,_,_,ss = get_score(predict_label_and_marked_label_list)
                print(score)
                csv_line.append(str(score))
                # print(ss)
            else:
                csv_line.append('')
            f = open(outfile, 'a+', encoding='utf-8')
            writer = csv.writer(f)
            writer.writerow(csv_line)
            f.close()



if __name__ == '__main__':
    fire.Fire()