#!/usr/bin/env python
#coding=utf8


import csv 
import math
import sys
import pickle

def compute_position_weighted_precision(correct_num, predict_num):
    assert(len(correct_num) == len(predict_num))
    weighted_correct = 0.0
    weighted_predict = 0.0
    for i in range(len(correct_num)):
        weighted_correct += correct_num[i] / math.log(i + 3.0)
        weighted_predict += predict_num[i] / math.log(i + 3.0)
    return weighted_correct / weighted_predict

def compute_recall(correct_num, ground_truth_num):
    return sum(correct_num) / ground_truth_num

def eval(ground_truth_data, predict_data, savepath, labelNumStr, max_tag_num=5):
    labelNum = int(labelNumStr) + 1
    label_truth = [0 for i in range(labelNum)]
    label_right = [0 for i in range(labelNum)]
    right_num = 0
    label_false = [0 for i in range(labelNum)]
    false_num = 0
    ground_truth = {}
    ground_truth_num = 0.0
    with open(ground_truth_data, 'r') as f:
        lines = csv.reader(f)
        for i, items in enumerate(lines):
            if i == 0:
                continue
            id, true_tag_ids = items[0], items[4]
            ground_truth[id] = true_tag_ids.split('|')
            for labelid in ground_truth[id]:
                label_truth[int(labelid)] += 1
                ground_truth_num += 1
            # ground_truth_num += len(ground_truth[id])
            # import ipdb;ipdb.set_trace()

    correct_num = [0.0] * max_tag_num
    predict_num = [0.0] * max_tag_num

    with open(predict_data, 'r') as f:
        lines = csv.reader(f)
        for i, items in enumerate(lines):
            # import ipdb;ipdb.set_trace()
            if i == 0:
                continue
            assert(len(items) == max_tag_num + 1)
            id = items[0]
            if id not in ground_truth:
                print('error id %d' %id)
                continue
            #assert(id in ground_truth)
            true_tag_ids = ground_truth[id]
            for pos, tag_id in enumerate(items[1:]):
                if tag_id == '-1':
                    continue
                if tag_id in true_tag_ids:
                    right_num += 1
                    label_right[int(tag_id)] += 1
                else:
                    false_num += 1
                    label_false[int(tag_id)] += 1

                predict_num[pos] += 1
                if tag_id in true_tag_ids:
                    correct_num[pos] += 1
    print(i)
    # 是否需要加一
    print(label_right)
    print(right_num)
    print(false_num)
    for i in range(labelNum):
        label_right[i] /= right_num
        label_false[i] /= false_num
    # print(savepath)
    spf = open(savepath, 'wb')
    pickle.dump({'r':label_right, 'f':label_false}, spf)

    precision = compute_position_weighted_precision(correct_num, predict_num)
    recall = compute_recall(correct_num, ground_truth_num)
    F1 = 2 * precision * recall / (precision + recall)

    print("precision: {}, recall: {}, F1 {}".format(precision, recall, F1))

    return F1

if __name__=='__main__':
    def usage():
        print("python %s dev_data.csv/test_data.csv result.csv" % sys.argv[0])
        exit(1)
    print(sys.argv)
    eval(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])