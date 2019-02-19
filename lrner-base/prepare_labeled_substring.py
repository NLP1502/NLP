# -*- coding: utf-8 -*-
from functools import reduce
import csv
import os

def get_sentence(file_url):
    with open(file_url) as f:
        data = f.read().split('\n')
    sentences = []
    star =0
    for end in range(len(data)):
        if data[end] == '':
            sentences.append(data[star:end])
            star = end + 1
    return sentences

label_list = []

def get_lable_name(label_id):
    return  label_list[label_id]

def get_sentence_raw(sentence):
    raw_num = len(sentence[0].strip().split())
    raw = [[] for _ in range(raw_num)]
    for i in range(len(sentence)):
        line = sentence[i].strip().split()
        for j in range(raw_num):
            raw[j].append(line[j])
    return raw

def get_label_ids(label_raw, begin_idx, end_idx, threshold=0):
    labels = []
    if threshold == 0:
        for i in range(begin_idx, end_idx):
            label = label_raw[i]
            if label != 'O':
                label = label.split('-')[1]
            if label not in label_list:
                label_list.append(label)
            label_id = label_list.index(label)
            if label_id not in labels:
                labels.append(label_id)
    elif threshold == 1:
        print 'get_label_idx not support threshold %d' % threshold
    else:
        print 'get_label_idx not support threshold %f' % threshold
    return labels

def control(in_file, out_dirct, feature_name, threshold=0, mode='one-class', combin_num=1,
            len_lower=1, len_upper=7, stop_at_punctuation = False, add_bpe = False, add_word_level = False):
    substrings = []
    feature_name_num = len(feature_name)
    in_data = get_sentence(in_file)
    in_file_name = os.path.split(in_file)[1]


    for lenth in range(len_lower, len_upper+1):
        out_data = [[] for _ in range(feature_name_num)]
        sentence_raw = []
        for sentence in in_data:
            if len(sentence) == 0:
                continue
            sentence_raw = get_sentence_raw(sentence)
            if mode == 'one-class':
                for begin_idx in range(0, len(sentence)-lenth+1):
                    label_ids = get_label_ids(sentence_raw[-1], begin_idx, begin_idx+lenth, threshold)
                    for feature in range(feature_name_num):
                        string = sentence_raw[feature][begin_idx : begin_idx+lenth]
                        out_data[feature].append(' '.join(string) + '\t' + ' '.join( map(get_lable_name, label_ids)))

            elif mode == 'multi-class':
                print 'multi-class not support now'

        for i in range(feature_name_num):
            out_file_name = in_file_name + '.' + feature_name[i] + '.len' + str(lenth)
            out_file_path = os.path.join(out_dirct, out_file_name)
            print 'writing to %s'%out_file_path
            with open(out_file_path,'w') as f:
                for l in range(len(out_data[i])):
                    f.write(out_data[i][l]+ '\n')

def get_char_from_word():
    pass

def get_bpe_from_word():
    pass

if __name__ == '__main__':
    feature_name = ['word', 'pos']
    # 1
    # out_dirct = 'data/substring/'
    # train_file = 'data/ned.train'
    # dev_file = 'data/ned.testa'
    # test_file = 'data/ned.testb'

    # 2
    # out_dirct = 'data/substring/'
    # train_file = 'data/cor.16.train.f'
    # dev_file = ''
    # test_file = 'data/cor.16.test.f'

    out_dirct = 'data/substring/'
    train_file = ''
    dev_file = ''
    test_file = 'data/ned.testb'

    # default: NER tag at last
    in_file = train_file
    if in_file != '':
        substrings = control(in_file, out_dirct, feature_name, threshold=0,
                         mode='one-class', combin_num=1, len_lower=1, len_upper=7)
    in_file = dev_file
    if in_file != '':
        substrings = control(in_file, out_dirct, feature_name, threshold=0,
                         mode='one-class', combin_num=1, len_lower=1, len_upper=7)
    in_file = test_file
    if in_file != '':
        substrings = control(in_file, out_dirct, feature_name, threshold=0,
                         mode='one-class', combin_num=1, len_lower=1, len_upper=7)

    get_char_from_word()
    get_bpe_from_word()

    print 'finish.'