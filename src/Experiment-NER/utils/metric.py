# -*- coding: utf-8 -*-

# from operator import add
# 
import numpy as np
import math
import sys
import os

# input as sentence level labels
def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES", show_detail=True):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):

        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]

        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)

        if label_type == "BMES":
            # BIOES tag scheme
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            # BIO tag scheme
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))  # 交集 ['[1,3]PER']
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner

    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    accuracy = (right_tag + 0.0) / all_tag
    # print entity info
    if show_detail:
        print "gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num
        get_entity_fmeasure([golden_full, predict_full, right_full])
    return accuracy, precision, recall, f_measure

def get_entity_fmeasure(full_list):
    # golden_full, predict_full, right_full
    ner_counts = []
    for full in full_list:
        entity = map(lambda x:x.split(']')[1], full)
        ner_count = {w:entity.count(w) for w in set(entity)}
        ner_counts.append(ner_count)

    precisions, recalls, f_measures = {}, {}, {}
    for key, golden_num in ner_counts[0].items():

        precision, recall, f_measure = -1, -1, -1
        pred_num = ner_counts[1][key] if key in ner_counts[1] else 0
        right_num = ner_counts[2][key] if key in ner_counts[2] else 0
        if pred_num != 0:
            precision = (right_num + 0.0) / pred_num
        recall = (right_num + 0.0) / golden_num
        if not ((precision == -1) or (recall == -1) or (precision + recall) <= 0.):
            f_measure = 2 * precision * recall / (precision + recall)

        precisions[key] = precision
        recalls[key] = recall
        f_measures[key] = f_measure
        print('\t%s: p: %.4f, r: %.4f, f: %.4f, %d, %d, %d'%(key, precision, recall, f_measure,
                                                          golden_num, pred_num, right_num))
    return precisions, recalls, f_measures

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    label_list = ["S-PER"]
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def get_ner_BIO(label_list):
    """
    example : label_list-->[O,B-PER,I-PER,O]
    :param label_list:
    :return:
    """

    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def readSentence(input_file):
    in_lines = open(input_file, 'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences, labels


def readTwoLabelSentence(input_file, pred_col=-1):
    in_lines = open(input_file, 'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if "##score##" in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[1])
            predict_label.append(pair[pred_col])

    return sentences, golden_labels, predict_labels


def fmeasure_from_file(golden_file, predict_file, label_type="BMES"):
    print "Get f measure from file:", golden_file, predict_file
    print "Label format:", label_type
    golden_sent, golden_labels = readSentence(golden_file)
    predict_sent, predict_labels = readSentence(predict_file)
    P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print ("P:%sm R:%s, F:%s" % (P, R, F))


def fmeasure_from_singlefile(twolabel_file, label_type="BMES", pred_col=-1):
    sent, golden_labels, predict_labels = readTwoLabelSentence(twolabel_file, pred_col)
    P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print ("P:%s, R:%s, F:%s" % (P, R, F))


if __name__ == '__main__':
    # print "sys:",len(sys.argv)
    if len(sys.argv) == 3:
        fmeasure_from_singlefile(sys.argv[1], "BMES", int(sys.argv[2]))
    else:
        fmeasure_from_singlefile(sys.argv[1], "BMES")
