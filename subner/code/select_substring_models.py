# -*- coding: utf-8 -*-
import shutil
import os
import pickle

# 对scorefilename里记录的模型重新筛选，把选中的存到另一目录，且重构记录的格式

scorefilename = '../data-middle/ned/test.word-pos.score_dict.pkl'
infiledir = '../data-substring-checkpoints/ned/word-pos/'
outfiledir = '../data-substring-checkpoints/ned/word-pos-selected/'
recordname = '../data-middle/ned/test.word-pos-selected.score_dict.pkl'

# 保留p\r\f各排前5的模型
select_num = 5

with open(scorefilename, 'rb') as fpkl:
    init_score_dict = pickle.load(fpkl)

classifer_count_dic ={}
for k, v in init_score_dict.items():
    name_split = k.split('.')
    feature_name = name_split[2].split('len')[0]
    feature_length = int(name_split[2].split('len')[1])
    feature_ans = [int(_) for _ in name_split[3][3:].split('-')]
    feature_ans_set = (feature_name, feature_length, tuple(sorted(feature_ans)))
    print feature_ans_set
    feature_score = float('0.' + name_split[4][5:])
    if feature_ans_set in classifer_count_dic:
        classifer_count_dic[feature_ans_set].append([k, v[1], v[2], v[3]])
    else:
        classifer_count_dic[feature_ans_set] = []
        classifer_count_dic[feature_ans_set].append([k, v[1], v[2], v[3]])

select_classifer_dic = {}
for k, v in classifer_count_dic.items():
    print len(v)
    print len(select_classifer_dic.items())
    p_s = sorted(v, key=lambda k: k[1], reverse=True)
    for i in range(select_num):
        if i == len(v):
            break
        select_classifer_dic[tuple(p_s[i])] = 0
    r_s = sorted(v, key=lambda k: k[2], reverse=True)
    for i in range(select_num):
        if i == len(v):
            break
        select_classifer_dic[tuple(r_s[i])] = 0
    f_s = sorted(v, key=lambda k: k[3], reverse=True)
    for i in range(select_num):
        if i == len(v):
            break
        select_classifer_dic[tuple(f_s[i])] = 0

for k, v in select_classifer_dic.items():
    name = k[0]
    shutil.move(os.path.join(infiledir,name), os.path.join(outfiledir,name))

with open(recordname, 'wb') as fpkl:
    pickle.dump(classifer_count_dic, fpkl, True)




