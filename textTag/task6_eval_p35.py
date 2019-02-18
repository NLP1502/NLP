#!/usr/bin/env python
#coding=utf8


import csv 
import math
import sys

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

def eval(ground_truth_data, predict_data, max_tag_num=5):
    ground_truth = {}
    ground_truth_num = 0.0
    with open(ground_truth_data, 'r') as f:
        lines = csv.reader(f)
        for i, items in enumerate(lines):
            if i == 0:
                continue
            id, true_tag_ids = items[0], items[4]
            ground_truth[id] = true_tag_ids.split('|')
            ground_truth_num += len(ground_truth[id])
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
                continue
            #assert(id in ground_truth)
            true_tag_ids = ground_truth[id]
            for pos, tag_id in enumerate(items[1:]):
                if tag_id == '-1':
                    continue
                predict_num[pos] += 1
                if tag_id in true_tag_ids:
                    correct_num[pos] += 1
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
    eval(sys.argv[1], sys.argv[2])