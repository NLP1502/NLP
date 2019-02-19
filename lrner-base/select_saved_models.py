import shutil
import re
import pickle

scorefilename = 'out.classification_scores'
infiledir = 'save_checked/'
outfiledir = 'save-substring/'
recordname = 'record.pkl'
with open(scorefilename, 'r') as f:
    score_of_models = []
    data = f.readlines()
    find = 0
    num = 0
    for dataline in data:
        if find == 2:
            kkk = re.split('[:,]',dataline.strip())
            if len(kkk) < 10:
                break
            p = float(kkk[-5])
            r = float(kkk[-3])
            f = float(kkk[-1])
            print('{},{},{},{}'.format(name, p, r, f))
            score_of_models.append([name, p, r, f])
            # shutil.move(infiledir+name, outfiledir + name)
            find = 1
            continue
        if find == 1:
            find = 0
            continue
        if dataline.strip().split(':')[0] == 'checkpiont':
            name = dataline.strip().split(':')[1].strip()
            find = 2
            num += 1
    print(num)

    classifer_count_dic = {}
    for record in score_of_models:
        name_split = record[0].split('.')
        feature_name = name_split[2].split('len')[0]
        feature_length = int(name_split[2].split('len')[1])
        feature_ans = [int(_) for _ in name_split[3][3:].split('-')]
        feature_ans_set = (feature_name, feature_length, tuple(sorted(feature_ans)))
        print feature_ans_set
        feature_score = float('0.'+name_split[4][5:])
        if feature_ans_set in classifer_count_dic:
            classifer_count_dic[feature_ans_set].append(record)
        else:
            classifer_count_dic[feature_ans_set] = []
            classifer_count_dic[feature_ans_set].append(record)

    select_num = 5
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
        shutil.move(infiledir+name, outfiledir + name)

    with open(outfiledir + recordname, 'wb') as fpkl:
        pickle.dump(classifer_count_dic, fpkl, True)




