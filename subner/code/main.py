# -*- coding: utf-8 -*-


import time
import sys
import argparse
import random
import copy
import torch
import gc
import cPickle as pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqmodel import SeqModel
from model.seqmodel import SeqModel_circulationBiLSTM
from model.LSTMText import LSTMText
from model.LSTMText_mix import LSTMText_mix
from utils.data import Data
from utils.data import init_parser
import os
from math import log, exp
from utils.get_score_classification import get_score_one
from dynamicPlanning import DynamicPlanning



def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        将预测的id恢复为label
        input:
            pred_variable (batch_size, sent_len): pred tag result tag_seq
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]

    # batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)

    # Variable to numpy()
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()

    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if
                         mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


# def save_data_setting(data, save_file):
#     new_data = copy.deepcopy(data)
#     ## remove input instances
#     new_data.train_texts = []
#     new_data.dev_texts = []
#     new_data.test_texts = []
#     new_data.raw_texts = []

#     new_data.train_Ids = []
#     new_data.dev_Ids = []
#     new_data.test_Ids = []
#     new_data.raw_Ids = []
#     ## save data settings
#     with open(save_file, 'w') as fp:
#         pickle.dump(new_data, fp)
#     print "Data setting saved to file: ", save_file


# def load_data_setting(save_file):
#     with open(save_file, 'r') as fp:
#         data = pickle.load(fp)
#     print "Data setting loaded from file: ", save_file
#     data.show_data_summary()
#     return data

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print "Learning rate is setted as:", lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def evaluate_nbest_rules_guide(data, model, rules_guide_predict, rules_guide_score, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print "Error: wrong evaluate name,", name

    show_nbest = True

    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []

    pred_results = []  # total pred result
    gold_results = []  # total gold result

    # set model in eval model
    model.eval()
    batch_size = data.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_trans, trans_seq_lengths, trans_seq_recover, mask = batchify_with_label(
            instance, data.gpu, False)


        if nbest and nbest >= 2:
            scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char,
                                                       batch_charlen, batch_charrecover, mask, nbest, batch_trans,
                                                       trans_seq_lengths, trans_seq_recover)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:, :, 0]

        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                            mask, batch_trans, trans_seq_lengths, trans_seq_recover)
        # print "tag:", tag_seq

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label

    # show nbest out
    instance_count = len(pred_scores)
    print('instance_count: {}'.format(instance_count))
    rules_guide_predict_count = len(rules_guide_predict)
    print('rules_guide_predict_count: {}'.format(rules_guide_predict_count))

    def substring_label_id_to_label_name(label_id):
        label_names = []
        for id in label_id:

            if id == 0:
                print('unk id in substring_label_id')
            else:
                label_names.append(data.substring_label_alphabet.instances[id-1])

        return label_names

    def label_strformat_to_substring_label(label):
        substring_label = []
        for lb in label:
            lbs = lb.split('-')
            substring_label.append(lbs[-1])
        return substring_label

    def get_rules_guide_score(in_label, rules_guide_score, substring_len):
        score = 0
        ## only for len1
        if substring_len != 1:
            print('not support for substraing_len which bigger than 1')
        in_label_trans = label_strformat_to_substring_label(in_label)
        in_label_trans_id = []
        for lbt in in_label_trans:
            in_label_trans_id.append(data.substring_label_alphabet.instance2index[lbt])
        for idx in range(0, len(in_label_trans)-substring_len+1):
            use_log_score = False
            if use_log_score:
                score += exp(rules_guide_score[idx][in_label_trans_id[idx]])
            else:
                score += rules_guide_score[idx][in_label_trans_id[idx]]
        return score

    # reranking
    pred_results_rerank = []
    for tn in range(train_num):
        if (tn+1) % 1000 == 0:
            print '%d'%(tn+1)
        substring_len1_score_list = []
        for j in range(nbest):
            substring_len1_score = get_rules_guide_score \
                (nbest_pred_results[tn][j], rules_guide_score[tn], 1)
            substring_len1_score_list.append(substring_len1_score)
        idnbest = substring_len1_score_list.index(max(substring_len1_score_list))
        pred_results_rerank.append(nbest_pred_results[tn][idnbest])

    if show_nbest:
        for i in range(200):
            x = random.randint(0, instance_count-1)
            if ','.join(gold_results[x]) == ','.join(nbest_pred_results[x][0]):
                continue
            print('---' * 10)
            print 'gold:       ' + ','.join(gold_results[x])
            for j in range(nbest):
                substring_len1_score = get_rules_guide_score\
                    (nbest_pred_results[x][j], rules_guide_score[x], 1)
                print '%.8f: '%(pred_scores[x][j]) + \
                      ','.join(nbest_pred_results[x][j]) + '\ts1:%.4f'%(substring_len1_score)

            rules_guide_predict_new = substring_label_id_to_label_name(rules_guide_predict[x])
            # print('{}'.format(rules_guide_predict[x]))

            print('ruleguide {}'.format(','.join(rules_guide_predict_new)))
            print('    O      ORG      MISC      PER      LOC')
            for idrgs, rgs in enumerate(rules_guide_score[x]):
                if gold_results[x][idrgs] != nbest_pred_results[x][0][idrgs]:
                    # print('{}'.format(rgs))
                    print('%.4f  ,%.4f  ,%.4f  ,%.4f  ,%.4f'%(rgs[1], rgs[2], rgs[3], rgs[4], rgs[5]))

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    print('*' * 10)
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    print("acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (acc, p, r, f))

    print('*' * 10)
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results_rerank, data.tagScheme)
    print("acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (acc, p, r, f))

    if nbest:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores

def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print "Error: wrong evaluate name,", name

    show_nbest = False

    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []

    pred_results = []  # total pred result
    gold_results = []  # total gold result

    # set model in eval model
    model.eval()
    batch_size = data.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_trans, trans_seq_lengths, trans_seq_recover, mask = batchify_with_label(
            instance, data.gpu, False)


        if nbest and nbest >= 2:
            scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char,
                                                       batch_charlen, batch_charrecover, mask, nbest, batch_trans,
                                                       trans_seq_lengths, trans_seq_recover)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:, :, 0]

        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                            mask, batch_trans, trans_seq_lengths, trans_seq_recover)
        # print "tag:", tag_seq

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label

    # show nbest out
    instance_count = len(pred_scores)
    if show_nbest:
        for i in range(10):
            x = random.randint(0, instance_count-1)
            print('---' * 10)
            print 'gold:       ' + ','.join(gold_results[x])
            for j in range(nbest):
                print '%.8f: '%(pred_scores[x][j]) + ','.join(nbest_pred_results[x][j])

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if nbest:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores



def dynamic_planning(feats_lists, inters):
    START_TAG = -2
    STOP_TAG = -1
    len_tag = 12
    ans_lists = []
    count = 1

    for feats in feats_lists:
        if count%1000 == 0:
            print(count)
        count += 1
        score = []
        score_path = []
        back_path = []
        score.append([inters[START_TAG][k] + feats[0][k] for k in range(len_tag)])
        score_path.append([k for k in range(len_tag)])
        for i in range(1, len(feats)):
            # score.append([max(score[-1][j] + inters[k][j] + feats[i][k] for j in range(len_tag)) for k in range(len_tag)])
            st = []
            stkpath = []
            for k in range(len_tag):
                stk = []
                for j in range(len_tag):
                    stk.append(score[-1][j] + inters[j][k] + feats[i][k])
                stkmax = max(stk)
                st.append(stkmax)
                stkpath.append(stk.index(stkmax))
            score.append(st)
            score_path.append(stkpath)
        score_all = [score[-1][j] + inters[j][STOP_TAG] for j in range(len_tag)]
        highest = max(score_all)
        highest_path = score_all.index(highest)
        back_path.append(highest_path)
        for back in range(len(feats)-1, -1, -1):
            back_path.append(score_path[back][back_path[-1]])
        back_path = back_path[:-1]
        back_path.reverse()
        ans_lists.append(back_path)
    return ans_lists

def evaluate_output_intermediate_result(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print "Error: wrong evaluate name,", name

    show_nbest = False

    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []

    pred_results = []  # total pred result
    gold_results = []  # total gold result

    # set model in eval model
    model.eval()
    batch_size = data.batch_size

    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    temp_start = time.time()
    feats_lists = []
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_trans, trans_seq_lengths, trans_seq_recover, mask = batchify_with_label(
            instance, data.gpu, False)


        if nbest and nbest >= 2:
            # scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char,
            #                                            batch_charlen, batch_charrecover, mask, nbest, batch_trans,
            #                                            trans_seq_lengths, trans_seq_recover)
            # nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            # nbest_pred_results += nbest_pred_result
            # pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            # ## select the best sequence to evalurate
            # tag_seq = nbest_tag_seq[:, :, 0]
            pass

        else:
            feats, trans = model.decode_output_intermediate_result(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                            mask, batch_trans, trans_seq_lengths, trans_seq_recover)
            out_feats = feats[batch_wordrecover]
            out_mask = mask[batch_wordrecover]
            for i in range(feats.shape[0]):
                feats_lists.append(out_feats[i][0:len(instance[i][0])].cpu().detach().numpy().tolist())
                # feats_list.append(out_feats[i][0:len(instance[i][0])])

            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                            mask, batch_trans, trans_seq_lengths, trans_seq_recover)

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    temp_time = time.time()
    temp_cost = temp_time - temp_start
    print('get middle Time: %.2fs' % temp_cost)

    # return feats_list, trans
    inters = trans.cpu().detach().numpy().tolist()

    return feats_lists, inters, gold_results, pred_results

def evaluate_classification_mix(data, model, name, feature_name, feature_length, feature_ans):
    if feature_name != 'word-pos':
        print('this vertion only for word-pos')
    feature_name_id_word = data.substring_names.index('word')
    feature_name_id_pos = data.substring_names.index('pos')
    class_num = 2

    if name == "train":
        instances_word = data.substring_train_Ids[feature_name_id_word][feature_length]
        instances_pos = data.substring_train_Ids[feature_name_id_pos][feature_length]
    elif name == "dev":
        instances_word = data.substring_dev_Ids[feature_name_id_word][feature_length]
        instances_pos = data.substring_dev_Ids[feature_name_id_pos][feature_length]
    elif name == 'test':
        instances_word = data.substring_test_Ids[feature_name_id_word][feature_length]
        instances_pos = data.substring_test_Ids[feature_name_id_pos][feature_length]
    else:
        print "Error: wrong evaluate name,", name

    instances = [[[[instances_word[indata_idx][0][len_idx], instances_pos[indata_idx][0][len_idx]]
                for len_idx in range(len(instances_word[indata_idx][0]))],
               instances_word[indata_idx][1]]
              for indata_idx in range(len(instances_word))]

    right_token = 0
    whole_token = 0

    nbest_pred_results = []
    pred_scores = []

    pred_results = []  # total pred result
    gold_results = []  # total gold result

    # set model in eval model
    model.eval()
    batch_size = data.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    pt = 0
    nt = 0
    pf = 0
    nf = 0
    b = 0.00000001
    temp_start = time.time()

    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue

        batch_word, batch_wordlen, batch_wordrecover, batch_label, mask = \
            batchify_with_label_classification_mix(instance, class_num, feature_ans, data.gpu)
        score = model(batch_word, batch_wordlen, mask)

        predict = score.data.topk(1, dim=1)[1].cpu().tolist()
        true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

        predict_label_and_marked_label_list = []
        for jj in range(batch_label.size(0)):
            if jj < end - start:
                predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))

        for pre, tru in predict_label_and_marked_label_list:
            if tru[0] == 0:
                if pre[0] == 0:
                    nf += 1
                else:
                    pf += 1
            else:
                if pre[0] == 0:
                    nt += 1
                else:
                    pt += 1

    p = pt / (pt + pf + b)
    r = pt / (pt + nt + b)
    a = (pt + nf) / (len(instances) + b)
    f = 2 * p * r / (p + r + b)

    temp_time = time.time()
    temp_cost = temp_time - temp_start
    speed =  train_num / temp_cost
    return speed, a, p, r, f

def evaluate_classification(data, model, name, feature_name, feature_length, feature_ans):
    feature_name_id = data.substring_names.index(feature_name)
    class_num = 2

    if name == "train":
        instances = data.substring_train_Ids[feature_name_id][feature_length]
    elif name == "dev":
        instances = data.substring_dev_Ids[feature_name_id][feature_length]
    elif name == 'test':
        instances = data.substring_test_Ids[feature_name_id][feature_length]
    else:
        print "Error: wrong evaluate name,", name

    right_token = 0
    whole_token = 0

    nbest_pred_results = []
    pred_scores = []

    pred_results = []  # total pred result
    gold_results = []  # total gold result

    # set model in eval model
    model.eval()
    batch_size = data.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    pt = 0
    nt = 0
    pf = 0
    nf = 0
    b = 0.00000001
    temp_start = time.time()

    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue

        batch_word, batch_wordlen, batch_wordrecover, batch_label, mask = \
            batchify_with_label_classification(instance, class_num, feature_ans, data.gpu)
        score = model(batch_word, batch_wordlen, mask)

        predict = score.data.topk(1, dim=1)[1].cpu().tolist()
        true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

        predict_label_and_marked_label_list = []
        for jj in range(batch_label.size(0)):
            if jj < end - start:
                predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))

        for pre, tru in predict_label_and_marked_label_list:
            if tru[0] == 0:
                if pre[0] == 0:
                    nf += 1
                else:
                    pf += 1
            else:
                if pre[0] == 0:
                    nt += 1
                else:
                    pt += 1

    p = pt / (pt + pf + b)
    r = pt / (pt + nt + b)
    a = (pt + nf) / (len(instances) + b)
    f = 2 * p * r / (p + r + b)

    temp_time = time.time()
    temp_cost = temp_time - temp_start
    speed =  train_num / temp_cost
    return speed, a, p, r, f

def batchify_with_label_classification(instance, label_class, feature_ans, gpu, requires_grad_flag=True):
    batch_size = len(instance)
    words = [sent[0] for sent in instance]
    labels = [sent[1] for sent in instance]
    batch_label_check = get_batch_label(labels, feature_ans)

    batch_label_check_t = torch.LongTensor(batch_label_check).view(-1, 1)
    batch_label_check_onehot = torch.zeros(batch_size, 2).long().scatter_(1, batch_label_check_t, 1)

    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max().numpy().item()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, label_class)), requires_grad=requires_grad_flag).long()

    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, batch_label_check_onehot, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :label_class] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen.numpy().item())

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, label_seq_tensor, mask

def batchify_with_label_classification_mix(instance, label_class, feature_ans, gpu, requires_grad_flag=True):
    batch_size = len(instance)
    words = [sent[0] for sent in instance]
    labels = [sent[1] for sent in instance]
    batch_label_check = get_batch_label(labels, feature_ans)

    batch_label_check_t = torch.LongTensor(batch_label_check).view(-1, 1)
    batch_label_check_onehot = torch.zeros(batch_size, 2).long().scatter_(1, batch_label_check_t, 1)

    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max().numpy().item()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, 2)), requires_grad=requires_grad_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, label_class)), requires_grad=requires_grad_flag).long()

    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, batch_label_check_onehot, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :label_class] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen.numpy().item())

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, label_seq_tensor, mask

def batchify_without_label_norank_classification(instance, gpu, requires_grad_flag=True):
    batch_size = len(instance)
    words = [sent[0] for sent in instance]

    word_seq_tensor = autograd.Variable(torch.Tensor(words), requires_grad=requires_grad_flag).long()

    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max().numpy().item()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).byte()

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, mask

def batchify_with_label_norank_classification(instance, label_class, feature_ans, gpu, requires_grad_flag=True):
    batch_size = len(instance)
    words = [sent[0] for sent in instance]
    labels = [sent[1] for sent in instance]
    batch_label_check = get_batch_label(labels, feature_ans)

    batch_label_check_t = torch.LongTensor(batch_label_check).view(-1, 1)
    batch_label_check_onehot = torch.zeros(batch_size, 2).long().scatter_(1, batch_label_check_t, 1)

    word_seq_tensor = autograd.Variable(torch.Tensor(words), requires_grad=requires_grad_flag).long()
    label_seq_tensor = torch.LongTensor(batch_label_check_onehot)

    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max().numpy().item()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).byte()

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, label_seq_tensor, mask
def batchify_with_label_norank_classification_mix(instance, label_class, feature_ans, gpu, requires_grad_flag=True):
    batch_size = len(instance)
    words = [sent[0] for sent in instance]
    labels = [sent[1] for sent in instance]
    batch_label_check = get_batch_label(labels, feature_ans)

    batch_label_check_t = torch.LongTensor(batch_label_check).view(-1, 1)
    batch_label_check_onehot = torch.zeros(batch_size, 2).long().scatter_(1, batch_label_check_t, 1)

    word_seq_tensor = autograd.Variable(torch.Tensor(words), requires_grad=requires_grad_flag).long()
    label_seq_tensor = torch.LongTensor(batch_label_check_onehot)

    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max().numpy().item()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).byte()

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, label_seq_tensor, mask

def batchify_with_label(input_batch_list, gpu, requires_grad_flag=True):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    # feature_num = 0
    chars = [sent[2] for sent in input_batch_list]
    trans = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max().numpy().item()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(
            autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).long())
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad=requires_grad_flag).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen.numpy().item())
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    ### deal with char
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [map(len, pad_char) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)),
                                        requires_grad=requires_grad_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    ### deal with trans
    pad_trans = [trans[idx] + [[0]] * (max_seq_len - len(trans[idx])) for idx in range(len(trans))]
    trans_length_list = [map(len, pdd_tran) for pdd_tran in pad_trans]
    max_tran_len = max(map(max, trans_length_list))
    trans_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_tran_len)),
                                         requires_grad=requires_grad_flag).long()
    trans_seq_lengths = torch.LongTensor(trans_length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_trans, trans_seq_lengths)):
        for idy, (tran, tranlen) in enumerate(zip(seq, seqlen)):
            trans_seq_tensor[idx, idy, :tranlen] = torch.LongTensor(tran)

    trans_seq_tensor = trans_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    trans_seq_lengths = trans_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    trans_seq_lengths, trans_perm_idx = trans_seq_lengths.sort(0, descending=True)
    trans_seq_tensor = trans_seq_tensor[trans_perm_idx]
    _, trans_seq_recover = trans_perm_idx.sort(0, descending=False)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        trans_seq_tensor = trans_seq_tensor.cuda()
        trans_seq_recover = trans_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, \
           char_seq_tensor, char_seq_lengths, char_seq_recover, \
           label_seq_tensor, \
           trans_seq_tensor, trans_seq_lengths, trans_seq_recover, mask


def train_circulationBiLSTM(data):
    print "Training model..."

    circul_time = data.circul_time
    deepth = data.circul_deepth
    model = SeqModel_circulationBiLSTM(data, circul_time, deepth)
    print "model:{}".format(model)

    if data.gpu:
        model.cuda()

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.SGD(model.word_hidden.wordrep.w.parameters(), lr=data.lr, momentum=data.momentum,
                                     weight_decay=data.l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.lr, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.Adam(model.word_hidden.wordrep.w.parameters(), lr=data.lr, weight_decay=data.l2)
    else:
        print("Optimizer illegal: %s , use sgd or adam." % data.optimizer)
        exit(0)

    best_dev = -10
    best_dev_epoch = -1
    best_test = -10
    best_test_epoch = -1
    # start training
    for idx in range(data.iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx + 1, data.iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)
            if data.use_mapping:
                optimizer_wc = lr_decay(optimizer_wc, idx, data.lr_decay, data.lr)

        instance_count = 0
        sample_id = 0
        #
        sample_loss = 0
        sample_mapping_loss = 0

        total_loss = 0
        total_mapping_loss = 0

        right_token = 0
        whole_token = 0

        random.shuffle(data.train_Ids)

        # set model in train mode
        model.train()
        model.zero_grad()
        batch_size = data.batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]

            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_trans, trans_seq_lengths, trans_seq_recover, mask = batchify_with_label(
                instance, data.gpu)
            instance_count += 1
            loss, tag_seq, wc_loss = model.neg_log_likelihood_loss(batch_word, batch_features, batch_wordlen,
                                                                   batch_char,
                                                                   batch_charlen, batch_charrecover, batch_label, mask,
                                                                   batch_trans, trans_seq_lengths, trans_seq_recover)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            #sample_loss += loss.data[0]
            sample_loss += loss.data.item()
            if data.use_mapping:
                sample_mapping_loss += wc_loss.data[0]

            #total_loss += loss.data[0]
            total_loss += loss.data.item()
            if data.use_mapping:
                total_mapping_loss += wc_loss.data[0]
            if batch_id % data.show_loss_per_batch == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                if data.use_mapping:
                    print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                        batch_id, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                else:
                    print("     Instance: %s; Time: %.2fs; loss: %.4f;mapping_loss: %.4f; acc: %s/%s=%.4f" % (
                        batch_id, temp_cost, sample_loss, sample_mapping_loss, right_token, whole_token,
                        (right_token + 0.) / whole_token))
                sys.stdout.flush()
                sample_loss = 0
                sample_mapping_loss = 0

            if data.use_trans and data.use_mapping:
                for param in model.word_hidden.wordrep.w.parameters():
                    param.requires_grad = False
                loss.backward(retain_graph=True)
                if data.clip != None:
                    torch.nn.utils.clip_grad_norm(model.parameters(), data.clip)
                optimizer.step()
                model.zero_grad()
                for param in model.word_hidden.wordrep.w.parameters():
                    param.requires_grad = True
                wc_loss.backward()
                optimizer_wc.step()
                model.zero_grad()
            else:
                loss.backward()
                if data.clip != None:
                    torch.nn.utils.clip_grad_norm(model.parameters(), data.clip)
                optimizer.step()
                model.zero_grad()

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        if data.use_mapping:
            print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                batch_id, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
        else:
            print("     Instance: %s; Time: %.2fs; loss: %.4f;mapping_loss: %.4f; acc: %s/%s=%.4f" % (
                batch_id, temp_cost, sample_loss, sample_mapping_loss, right_token, whole_token,
                (right_token + 0.) / whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        if data.use_mapping:
            print(
                    "Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s,total mapping loss: %s" % (
                idx + 1, epoch_cost, train_num / epoch_cost, total_loss, total_mapping_loss))
        else:
            print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
                idx + 1, epoch_cost, train_num / epoch_cost, total_loss))

        # continue
        speed, acc, p, r, f, _, _ = evaluate(data, model, "dev", data.nbest)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        if current_score > best_dev:
            if data.seg:
                print "Exceed previous best f score:", best_dev
            else:
                print "Exceed previous best acc score:", best_dev
            if data.save_model:
                model_name = data.model_dir + data.state_training_name + '.' + str(current_score)[2:-1]
                print "Save current best model in file:", model_name
                torch.save(model.state_dict(), model_name)
            best_dev = current_score
            best_dev_epoch = idx
            # ## decode test
        speed, acc, p, r, f, _, _ = evaluate(data, model, "test", data.nbest)
        if f > best_test:
            best_test = f
            best_test_epoch = idx

        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        print('best_dev_score: %.4f, best_dev_epoch:%d'%(best_dev,best_dev_epoch))
        print('best_test_score: %.4f, best_test_epoch:%d'%(best_test,best_test_epoch))
        gc.collect()

def classification_nbest_rules_guide_selected_models(data, name):
    print "Training model..."

    # model_path = 'save-train/'
    model_path = './'
    model_name = 'save-traindefault.87010229685'
    model = SeqModel(data)
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    print "model:{}".format(model)

    if data.gpu:
        model.cuda()

    out_predict, out_predict_score = make_classification_ans_ensemble_selected_models(data, 'test')

    speed, acc, p, r, f, _, _ = evaluate_nbest_rules_guide(data, model, out_predict, out_predict_score, name, data.nbest)

def classification_nbest_rules_guide(data, name):
    print "Training model..."

    # model_path = 'save-train/'
    model_path = './'
    model_name = 'save-traindefault.86302424009'
    model = SeqModel(data)
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    print "model:{}".format(model)

    if data.gpu:
        model.cuda()

    out_predict, out_predict_score = make_classification_ans_ensemble(data, 'test')

    speed, acc, p, r, f, _, _ = evaluate_nbest_rules_guide(data, model, out_predict, out_predict_score, name, data.nbest)


def substring_viterbi_prepare(data, name='test'):
    print "substring_viterbi_prepare ..."

    # model_path = 'save-train/'
    model_path = './'
    model_name = 'save-traindefault.8711916698'
    exp_len_max = 8
    get_init_score = True
    best_score = 0
    init_score = None

    model = SeqModel(data)
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    print "model:{}".format(model)

    if data.gpu:
        model.cuda()

    feats_lists, inters, gold_results, pred_results = evaluate_output_intermediate_result(data, model, name, data.nbest)
    pickle_save = [feats_lists, inters, gold_results, pred_results]

    f1 = file('substring_viterbi_prepare.pkl', 'wb')
    pickle.dump(pickle_save, f1, True)
    f1.close()

def substring_viterbi(data, name='test', searchmode='inturn',
                   prepared_data_path='substring_viterbi_prepare.pkl'):
    print "substring_viterbi ..."
    ans_dict_file_name = name + '.' + data.feature_name + '.ans_dict.pkl'
    ans_dict_file_path = os.path.join(data.middle_dir, data.task_name)
    ans_dict_file_path = os.path.join(ans_dict_file_path, ans_dict_file_name)
    f1 = file(ans_dict_file_path, 'rb')
    ans_dict = pickle.load(f1)
    f1.close()

    viterbi_inputs_path = os.path.join(data.middle_dir, data.task_name)
    viterbi_inputs_name = data.viterbi_inputs_model_name + '.viterbi_inputs.pkl'
    viterbi_inputs_path = os.path.join(viterbi_inputs_path, viterbi_inputs_name)
    f1 = file(viterbi_inputs_path, 'rb')
    [feats_lists, inters, gold_results, pred_results] = pickle.load(f1)
    f1.close()

    exp_len_max = 8
    get_init_score = True
    best_score = 0
    init_score = None


    # compare with init method
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    print('p:{}, r:{}, f:{}'.format(p,r,f))


    if get_init_score:
        print('get_init_score')
        temp_start = time.time()
        ans_lists = dynamic_planning(feats_lists, inters)
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print('dynamic_planning Time: %.2fs' % temp_cost)
        ans_lists_label = [[data.label_alphabet.get_instance(ans) for ans in ans_list] for ans_list in ans_lists]
        acc2, p2, r2, f2 = get_ner_fmeasure(gold_results, ans_lists_label, data.tagScheme)
        print('p:{}, r:{}, f:{}'.format(p2, r2, f2))
        init_score = f2
        best_score = f2

    if searchmode == 'example':
        myDynamicPlanning = DynamicPlanning()
        myDynamicPlanning.select_ans_dict(data, ans_dict, feats_lists, maxlen=exp_len_max - 1)
        temp_start = time.time()
        ans_lists = myDynamicPlanning.dynamic_planning_plus(data, feats_lists, inters, ans_dict)
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print('dynamic_planning_plus Time: %.2fs' % temp_cost)
        ans_lists_label = [[data.label_alphabet.get_instance(ans) for ans in ans_list] for ans_list in ans_lists]
        acc3, p3, r3, f3 = get_ner_fmeasure(gold_results, ans_lists_label, data.tagScheme)
        print('p:{}, r:{}, f:{}'.format(p3, r3, f3))
        print('add: {}; ad_len_weight {}'.format(f3 - init_score, myDynamicPlanning.ad_len_weight))
        print('*' * 10)


    if searchmode == 'inturn_small':
        myDynamicPlanning = DynamicPlanning()
        myDynamicPlanning.select_ans_dict(data, ans_dict, feats_lists, maxlen=exp_len_max - 1)
        for exp_len_pos in range(1,exp_len_max):
            exp_len_para = [0 for _ in range(exp_len_max)]
            for exp_len_pos_weight in [1.0/30, 1.0/40, 1.0/15, 1.0/1000, 1.0/100, 1.0/80, 1.0/60, 1.0/10, 1.0/5]:
                exp_len_para[exp_len_pos] = exp_len_pos_weight
                myDynamicPlanning.ad_len_weight = exp_len_para

                temp_start = time.time()
                ans_lists = myDynamicPlanning.dynamic_planning_plus(data, feats_lists, inters, ans_dict)
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                print('## experiment: ad_len_weight {}'.format(myDynamicPlanning.ad_len_weight))
                # print('## experiment: adp_prf_weight_para {}'.format(myDynamicPlanning.adp_prf_weight_para))
                # print('## experiment: ad_ans_num_weight_para {}'.format(myDynamicPlanning.ad_ans_num_weight_para))
                # print('## experiment: adp_scores_weight_para {}'.format(myDynamicPlanning.adp_scores_weight_para))
                print('dynamic_planning_plus Time: %.2fs' % temp_cost)
                ans_lists_label = [[data.label_alphabet.get_instance(ans) for ans in ans_list] for ans_list in ans_lists]
                acc3, p3, r3, f3 = get_ner_fmeasure(gold_results, ans_lists_label, data.tagScheme)
                print('p:{}, r:{}, f:{}'.format(p3, r3, f3))
                print('add: {}; ad_len_weight {}'.format(f3-init_score, myDynamicPlanning.ad_len_weight))
                print('*' * 10)

    if searchmode == 'inturn_big':
        myDynamicPlanning = DynamicPlanning()
        myDynamicPlanning.select_ans_dict(data, ans_dict, feats_lists, maxlen=exp_len_max-1)
        for exp_len_pos in range(1,exp_len_max):
            exp_len_para = [0 for _ in range(exp_len_max)]
            for exp_len_pos_weight in [1.0/30, 1.0/40, 1.0/15, 1.0/1000, 1.0/100, 1.0/80, 1.0/60, 1.0/10, 1.0/5]:
                exp_len_para[exp_len_pos] = exp_len_pos_weight
                myDynamicPlanning.ad_len_weight = exp_len_para
                for exp_adp_prf_weight_para in [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0]
                    , [1.0, 1.0, 0], [0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 0.5, 0], [1.0, 0, 0.5]]:
                    myDynamicPlanning.adp_prf_weight_para = exp_adp_prf_weight_para
                    for exp_ad_ans_num_weight_para in [1.0, 0.5, 0.0]:
                        myDynamicPlanning.ad_ans_num_weight_para = exp_ad_ans_num_weight_para
                        for exp_adp_scores_weight_para in [[1.0, 0], [0, 1.0]]:
                            myDynamicPlanning.adp_scores_weight_para = exp_adp_scores_weight_para

                            temp_start = time.time()
                            ans_lists = myDynamicPlanning.dynamic_planning_plus(data, feats_lists, inters, ans_dict)
                            temp_time = time.time()
                            temp_cost = temp_time - temp_start
                            print('## experiment: ad_len_weight {}'.format(myDynamicPlanning.ad_len_weight))
                            print('## experiment: adp_prf_weight_para {}'.format(myDynamicPlanning.adp_prf_weight_para))
                            print('## experiment: ad_ans_num_weight_para {}'.format(myDynamicPlanning.ad_ans_num_weight_para))
                            print('## experiment: adp_scores_weight_para {}'.format(myDynamicPlanning.adp_scores_weight_para))
                            print('dynamic_planning_plus Time: %.2fs' % temp_cost)
                            ans_lists_label = [[data.label_alphabet.get_instance(ans) for ans in ans_list] for ans_list in ans_lists]
                            acc3, p3, r3, f3 = get_ner_fmeasure(gold_results, ans_lists_label, data.tagScheme)
                            print('p:{}, r:{}, f:{}'.format(p3, r3, f3))

def decode_prepare(data, name='test', searchmode='inturn'):
    print "decode prepare ..."
    f1 = file(name + '.ans_dict.pkl', 'rb')
    ans_dict = pickle.load(f1)
    f1.close()

    # model_path = 'save-train/'
    model_path = './'
    model_name = 'save-traindefault.8711916698'
    exp_len_max = 8
    get_init_score = True
    best_score = 0
    init_score = None

    model = SeqModel(data)
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    print "model:{}".format(model)

    if data.gpu:
        model.cuda()

    feats_lists, inters, gold_results, pred_results = evaluate_output_intermediate_result(data, model, name, data.nbest)

    # compare with init method
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    print('p:{}, r:{}, f:{}'.format(p,r,f))


    if get_init_score:
        print('get_init_score')
        temp_start = time.time()
        ans_lists = dynamic_planning(feats_lists, inters)
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print('dynamic_planning Time: %.2fs' % temp_cost)
        ans_lists_label = [[data.label_alphabet.get_instance(ans) for ans in ans_list] for ans_list in ans_lists]
        acc2, p2, r2, f2 = get_ner_fmeasure(gold_results, ans_lists_label, data.tagScheme)
        print('p:{}, r:{}, f:{}'.format(p2, r2, f2))
        init_score = f2
        best_score = f2

    if searchmode == 'example':
        myDynamicPlanning = DynamicPlanning()
        myDynamicPlanning.select_ans_dict(data, ans_dict, feats_lists, maxlen=exp_len_max - 1)
        temp_start = time.time()
        ans_lists = myDynamicPlanning.dynamic_planning_plus(data, feats_lists, inters, ans_dict)
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print('dynamic_planning_plus Time: %.2fs' % temp_cost)
        ans_lists_label = [[data.label_alphabet.get_instance(ans) for ans in ans_list] for ans_list in ans_lists]
        acc3, p3, r3, f3 = get_ner_fmeasure(gold_results, ans_lists_label, data.tagScheme)
        print('p:{}, r:{}, f:{}'.format(p3, r3, f3))
        print('add: {}; ad_len_weight {}'.format(f3 - init_score, myDynamicPlanning.ad_len_weight))
        print('*' * 10)


    if searchmode == 'inturn_small':
        myDynamicPlanning = DynamicPlanning()
        myDynamicPlanning.select_ans_dict(data, ans_dict, feats_lists, maxlen=exp_len_max - 1)
        for exp_len_pos in range(1,exp_len_max):
            exp_len_para = [0 for _ in range(exp_len_max)]
            for exp_len_pos_weight in [1.0/30, 1.0/40, 1.0/15, 1.0/1000, 1.0/100, 1.0/80, 1.0/60, 1.0/10, 1.0/5]:
                exp_len_para[exp_len_pos] = exp_len_pos_weight
                myDynamicPlanning.ad_len_weight = exp_len_para

                temp_start = time.time()
                ans_lists = myDynamicPlanning.dynamic_planning_plus(data, feats_lists, inters, ans_dict)
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                print('## experiment: ad_len_weight {}'.format(myDynamicPlanning.ad_len_weight))
                # print('## experiment: adp_prf_weight_para {}'.format(myDynamicPlanning.adp_prf_weight_para))
                # print('## experiment: ad_ans_num_weight_para {}'.format(myDynamicPlanning.ad_ans_num_weight_para))
                # print('## experiment: adp_scores_weight_para {}'.format(myDynamicPlanning.adp_scores_weight_para))
                print('dynamic_planning_plus Time: %.2fs' % temp_cost)
                ans_lists_label = [[data.label_alphabet.get_instance(ans) for ans in ans_list] for ans_list in ans_lists]
                acc3, p3, r3, f3 = get_ner_fmeasure(gold_results, ans_lists_label, data.tagScheme)
                print('p:{}, r:{}, f:{}'.format(p3, r3, f3))
                print('add: {}; ad_len_weight {}'.format(f3-init_score, myDynamicPlanning.ad_len_weight))
                print('*' * 10)

    if searchmode == 'inturn_big':
        myDynamicPlanning = DynamicPlanning()
        myDynamicPlanning.select_ans_dict(data, ans_dict, feats_lists, maxlen=exp_len_max-1)
        for exp_len_pos in range(1,exp_len_max):
            exp_len_para = [0 for _ in range(exp_len_max)]
            for exp_len_pos_weight in [1.0/30, 1.0/40, 1.0/15, 1.0/1000, 1.0/100, 1.0/80, 1.0/60, 1.0/10, 1.0/5]:
                exp_len_para[exp_len_pos] = exp_len_pos_weight
                myDynamicPlanning.ad_len_weight = exp_len_para
                for exp_adp_prf_weight_para in [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0]
                    , [1.0, 1.0, 0], [0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 0.5, 0], [1.0, 0, 0.5]]:
                    myDynamicPlanning.adp_prf_weight_para = exp_adp_prf_weight_para
                    for exp_ad_ans_num_weight_para in [1.0, 0.5, 0.0]:
                        myDynamicPlanning.ad_ans_num_weight_para = exp_ad_ans_num_weight_para
                        for exp_adp_scores_weight_para in [[1.0, 0], [0, 1.0]]:
                            myDynamicPlanning.adp_scores_weight_para = exp_adp_scores_weight_para

                            temp_start = time.time()
                            ans_lists = myDynamicPlanning.dynamic_planning_plus(data, feats_lists, inters, ans_dict)
                            temp_time = time.time()
                            temp_cost = temp_time - temp_start
                            print('## experiment: ad_len_weight {}'.format(myDynamicPlanning.ad_len_weight))
                            print('## experiment: adp_prf_weight_para {}'.format(myDynamicPlanning.adp_prf_weight_para))
                            print('## experiment: ad_ans_num_weight_para {}'.format(myDynamicPlanning.ad_ans_num_weight_para))
                            print('## experiment: adp_scores_weight_para {}'.format(myDynamicPlanning.adp_scores_weight_para))
                            print('dynamic_planning_plus Time: %.2fs' % temp_cost)
                            ans_lists_label = [[data.label_alphabet.get_instance(ans) for ans in ans_list] for ans_list in ans_lists]
                            acc3, p3, r3, f3 = get_ner_fmeasure(gold_results, ans_lists_label, data.tagScheme)
                            print('p:{}, r:{}, f:{}'.format(p3, r3, f3))

def train(data):
    print "Training model..."

    model = SeqModel(data)
    print "model:{}".format(model)

    if data.gpu:
        model.cuda()

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.SGD(model.word_hidden.wordrep.w.parameters(), lr=data.lr, momentum=data.momentum,
                                     weight_decay=data.l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.lr, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.Adam(model.word_hidden.wordrep.w.parameters(), lr=data.lr, weight_decay=data.l2)
    else:
        print("Optimizer illegal: %s , use sgd or adam." % data.optimizer)
        exit(0)

    best_dev = -10
    best_dev_epoch = -1
    best_test = -10
    best_test_epoch = -1
    # start training
    for idx in range(data.iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx + 1, data.iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)
            if data.use_mapping:
                optimizer_wc = lr_decay(optimizer_wc, idx, data.lr_decay, data.lr)

        instance_count = 0
        sample_id = 0
        #
        sample_loss = 0
        sample_mapping_loss = 0

        total_loss = 0
        total_mapping_loss = 0

        right_token = 0
        whole_token = 0

        random.shuffle(data.train_Ids)

        # set model in train mode
        model.train()
        model.zero_grad()
        batch_size = data.batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]

            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_trans, trans_seq_lengths, trans_seq_recover, mask = batchify_with_label(
                instance, data.gpu)
            instance_count += 1
            loss, tag_seq, wc_loss = model.neg_log_likelihood_loss(batch_word, batch_features, batch_wordlen,
                                                                   batch_char,
                                                                   batch_charlen, batch_charrecover, batch_label, mask,
                                                                   batch_trans, trans_seq_lengths, trans_seq_recover)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            #sample_loss += loss.data[0]
            sample_loss += loss.data.item()
            if data.use_mapping:
                sample_mapping_loss += wc_loss.data[0]

            #total_loss += loss.data[0]
            total_loss += loss.data.item()
            if data.use_mapping:
                total_mapping_loss += wc_loss.data[0]
            if batch_id % data.show_loss_per_batch == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                if data.use_mapping:
                    print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                        batch_id, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                else:
                    print("     Instance: %s; Time: %.2fs; loss: %.4f;mapping_loss: %.4f; acc: %s/%s=%.4f" % (
                        batch_id, temp_cost, sample_loss, sample_mapping_loss, right_token, whole_token,
                        (right_token + 0.) / whole_token))
                sys.stdout.flush()
                sample_loss = 0
                sample_mapping_loss = 0

            if data.use_trans and data.use_mapping:
                for param in model.word_hidden.wordrep.w.parameters():
                    param.requires_grad = False
                loss.backward(retain_graph=True)
                if data.clip != None:
                    torch.nn.utils.clip_grad_norm(model.parameters(), data.clip)
                optimizer.step()
                model.zero_grad()
                for param in model.word_hidden.wordrep.w.parameters():
                    param.requires_grad = True
                wc_loss.backward()
                optimizer_wc.step()
                model.zero_grad()
            else:
                loss.backward()
                if data.clip != None:
                    torch.nn.utils.clip_grad_norm(model.parameters(), data.clip)
                optimizer.step()
                model.zero_grad()

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        if data.use_mapping:
            print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                batch_id, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
        else:
            print("     Instance: %s; Time: %.2fs; loss: %.4f;mapping_loss: %.4f; acc: %s/%s=%.4f" % (
                batch_id, temp_cost, sample_loss, sample_mapping_loss, right_token, whole_token,
                (right_token + 0.) / whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        if data.use_mapping:
            print(
                    "Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s,total mapping loss: %s" % (
                idx + 1, epoch_cost, train_num / epoch_cost, total_loss, total_mapping_loss))
        else:
            print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
                idx + 1, epoch_cost, train_num / epoch_cost, total_loss))

        # continue
        speed, acc, p, r, f, _, _ = evaluate(data, model, "dev", data.nbest)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        if current_score > best_dev:
            if data.seg:
                print "Exceed previous best f score:", best_dev
            else:
                print "Exceed previous best acc score:", best_dev
            if data.save_model:
                model_name = os.path.join(data.model_dir, data.state_training_name + '.' + str(current_score)[2:-1])
                print "Save current best model in file:", model_name
                torch.save(model.state_dict(), model_name)
            best_dev = current_score
            best_dev_epoch = idx
            # ## decode test
        speed, acc, p, r, f, _, _ = evaluate(data, model, "test", data.nbest)
        if f > best_test:
            best_test = f
            best_test_epoch = idx

        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        print('best_dev_score: %.4f, best_dev_epoch:%d'%(best_dev,best_dev_epoch))
        print('best_test_score: %.4f, best_test_epoch:%d'%(best_test,best_test_epoch))
        gc.collect()

def get_batch_label(batch_label, feature_ans):
    batch_label_check = []

    for sent in batch_label:
        is_positive = 0
        for l in sent:
            if l in feature_ans:
                is_positive = 1

        batch_label_check.append(is_positive)

    return batch_label_check


def classification(data, feature_name, feature_length, class_num, feature_ans):
    print "Training classification..."

    # feature_name = 'word'
    # feature_length = 3
    # class_num = 2
    # feature_ans = [2,3]

    feature_ans_new = []
    for fa in feature_ans.split(','):
        feature_ans_new.append(data.substring_label_alphabet.instance2index[fa])
    feature_ans = feature_ans_new

    feature_name_id = data.substring_names.index(feature_name)

    model = LSTMText(data, feature_name, feature_ans, feature_length)
    print "model:{}".format(model)

    if data.gpu:
        model.cuda()

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.SGD(model.word_hidden.wordrep.w.parameters(), lr=data.lr, momentum=data.momentum,
                                     weight_decay=data.l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.lr, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.Adam(model.word_hidden.wordrep.w.parameters(), lr=data.lr, weight_decay=data.l2)
    else:
        print("Optimizer illegal: %s , use sgd or adam." % data.optimizer)
        exit(0)

    best_dev = -10
    best_dev_epoch = -1
    best_test = -10
    best_test_epoch = -1
    loss_function = torch.nn.MultiLabelSoftMarginLoss()
    # start training
    for idx in range(data.iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx + 1, data.iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)

        pt = 0
        nt = 0
        pf = 0
        nf = 0
        b = 0.00000001

        instance_count = 0

        sample_loss = 0
        sample_mapping_loss = 0

        total_loss = 0
        total_mapping_loss = 0

        right_token = 0
        whole_token = 0
        whole_token_per_check = 0

        random.shuffle(data.substring_train_Ids[feature_name_id][feature_length])

        # set model in train mode
        model.train()
        model.zero_grad()
        batch_size = data.batch_size
        batch_id = 0
        train_num = len(data.substring_train_Ids[feature_name_id][feature_length])
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            whole_token_per_check += end - start

            ### !!!!!!!!!!
            instance = data.substring_train_Ids[feature_name_id][feature_length][start:end]



            if not instance:
                continue
            label_class = 0
            batch_word, batch_wordlen, batch_wordrecover, batch_label, mask = \
                batchify_with_label_classification(instance, class_num, feature_ans, data.gpu)
            instance_count += 1
            score = model(batch_word, batch_wordlen, mask)

            weight = 1
            loss = loss_function(score, weight * batch_label.float())

            total_loss += loss.data

            predict = score.data.topk(1, dim=1)[1].cpu().tolist()
            true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

            predict_label_and_marked_label_list = []
            for jj in range(batch_label.size(0)):
                if jj < end - start:
                    predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))

            for pre, tru in predict_label_and_marked_label_list:
                if tru[0] == 0:
                    if pre[0] == 0:
                        nf += 1
                    else:
                        pf += 1
                else:
                    if pre[0] == 0:
                        nt += 1
                    else:
                        pt += 1

            if batch_id % data.show_loss_per_batch == 0:
                p = pt / (pt + pf + b)
                r = pt / (pt + nt + b)
                a = (pt + nf) / (whole_token_per_check + b)
                f = 2 * p * r / (p + r + b)
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; a: %.4f; p: %.4f; r: %.4f; f: %.4f" % (
                    batch_id, temp_cost,
                    a, p, r, f))
                whole_token_per_check = 0
                pt = 0
                nt = 0
                pf = 0
                nf = 0
                sys.stdout.flush()

            loss.backward()
            if data.clip != None:
                torch.nn.utils.clip_grad_norm(model.parameters(), data.clip)
            optimizer.step()
            model.zero_grad()

        temp_time = time.time()
        temp_cost = temp_time - temp_start

        print("     Instance: %s; Time: %.2fs; a: %.4f; p: %.4f; r: %.4f; f: %.4f" % (
            batch_id, temp_cost,
            a, p, r, f))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start

        print(
                "Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx + 1, epoch_cost, train_num / epoch_cost, total_loss))


        # continue
        speed, acc, p, r, f = evaluate_classification(data, model, "dev",
                feature_name, feature_length, feature_ans)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish


        current_score = f
        print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        save_when_get_new_best_score_on_dev = False
        save_each_epoch = True

        if save_when_get_new_best_score_on_dev:
            if current_score > best_dev:
                if data.seg:
                    print "Exceed previous best f score:", best_dev
                else:
                    print "Exceed previous best acc score:", best_dev
                if data.save_model:
                    model_name = data.model_dir + data.state_training_name + '.'\
                                 + feature_name + 'len%d'%(feature_length) + '.ans'\
                                 + '-'.join([str(_) for _ in feature_ans]) + '.score' + str(current_score)[2:-1]
                    print "Save current best model in file:", model_name
                    torch.save(model.state_dict(), model_name)
                best_dev = current_score
                best_dev_epoch = idx

        if save_each_epoch:
            if current_score > best_dev:
                best_dev = current_score
                best_dev_epoch = idx
            if data.seg:
                print "Exceed previous best f score:", best_dev
            else:
                print "Exceed previous best acc score:", best_dev
            if data.save_model:
                model_name = data.model_dir + data.state_training_name + '.'\
                             + feature_name + 'len%d'%(feature_length) + '.ans'\
                             + '-'.join([str(_) for _ in feature_ans]) + '.score' + str(current_score)[2:-1]
                print "Save model at %d epoch in file:" % (idx), model_name
                torch.save(model.state_dict(), model_name)

        ## decode test
        speed, acc, p, r, f = evaluate_classification(data, model, "test",
            feature_name, feature_length, feature_ans)

        if f > best_test:
            best_test = f
            best_test_epoch = idx

        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        print('best_dev_score: %.4f, best_dev_epoch:%d'%(best_dev,best_dev_epoch))
        print('best_test_score: %.4f, best_test_epoch:%d'%(best_test,best_test_epoch))
        gc.collect()

def classification_mix(data, feature_name, feature_length, class_num, feature_ans):
    print "Training classification_mix..."

    # feature_name = 'word'
    # feature_length = 3
    # class_num = 2
    # feature_ans = [2,3]

    feature_ans_new = []
    for fa in feature_ans.split(','):
        feature_ans_new.append(data.substring_label_alphabet.instance2index[fa])
    feature_ans = feature_ans_new

    feature_name_id_word = data.substring_names.index('word')
    feature_name_id_pos = data.substring_names.index('pos')

    model = LSTMText_mix(data, feature_name, feature_ans, feature_length)
    print "model:{}".format(model)

    if data.gpu:
        model.cuda()

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.SGD(model.word_hidden.wordrep.w.parameters(), lr=data.lr, momentum=data.momentum,
                                     weight_decay=data.l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.lr, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.Adam(model.word_hidden.wordrep.w.parameters(), lr=data.lr, weight_decay=data.l2)
    else:
        print("Optimizer illegal: %s , use sgd or adam." % data.optimizer)
        exit(0)

    best_dev = -10
    best_dev_epoch = -1
    best_test = -10
    best_test_epoch = -1
    loss_function = torch.nn.MultiLabelSoftMarginLoss()
    # start training
    for idx in range(data.iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx + 1, data.iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)

        pt = 0
        nt = 0
        pf = 0
        nf = 0
        b = 0.00000001

        instance_count = 0

        sample_loss = 0
        sample_mapping_loss = 0

        total_loss = 0
        total_mapping_loss = 0

        right_token = 0
        whole_token = 0
        whole_token_per_check = 0

        indata_word = data.substring_train_Ids[feature_name_id_word][feature_length]
        indata_pos = data.substring_train_Ids[feature_name_id_pos][feature_length]
        # indata = [[indata_word[indata_idx][0], indata_pos[indata_idx][0], indata_word[indata_idx][1]]
        #           for indata_idx in range(len(indata_word))]
        indata = [[[[indata_word[indata_idx][0][len_idx], indata_pos[indata_idx][0][len_idx]]
                    for len_idx in range(len(indata_word[indata_idx][0]))],
                   indata_word[indata_idx][1]]
                  for indata_idx in range(len(indata_word))]

        random.shuffle(indata)

        # set model in train mode
        model.train()
        model.zero_grad()
        batch_size = data.batch_size
        batch_id = 0
        train_num = len(indata)
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            whole_token_per_check += end - start

            ### !!!!!!!!!!
            instance = indata[start:end]



            if not instance:
                continue
            label_class = 0
            batch_word, batch_wordlen, batch_wordrecover, batch_label, mask = \
                batchify_with_label_classification_mix(instance, class_num, feature_ans, data.gpu)
            instance_count += 1
            score = model(batch_word, batch_wordlen, mask)

            weight = 1
            loss = loss_function(score, weight * batch_label.float())

            total_loss += loss.data

            predict = score.data.topk(1, dim=1)[1].cpu().tolist()
            true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

            predict_label_and_marked_label_list = []
            for jj in range(batch_label.size(0)):
                if jj < end - start:
                    predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))

            for pre, tru in predict_label_and_marked_label_list:
                if tru[0] == 0:
                    if pre[0] == 0:
                        nf += 1
                    else:
                        pf += 1
                else:
                    if pre[0] == 0:
                        nt += 1
                    else:
                        pt += 1

            if batch_id % data.show_loss_per_batch == 0:
                p = pt / (pt + pf + b)
                r = pt / (pt + nt + b)
                a = (pt + nf) / (whole_token_per_check + b)
                f = 2 * p * r / (p + r + b)
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; a: %.4f; p: %.4f; r: %.4f; f: %.4f" % (
                    batch_id, temp_cost,
                    a, p, r, f))
                whole_token_per_check = 0
                pt = 0
                nt = 0
                pf = 0
                nf = 0
                sys.stdout.flush()

            loss.backward()
            if data.clip != None:
                torch.nn.utils.clip_grad_norm(model.parameters(), data.clip)
            optimizer.step()
            model.zero_grad()

        temp_time = time.time()
        temp_cost = temp_time - temp_start

        print("     Instance: %s; Time: %.2fs; a: %.4f; p: %.4f; r: %.4f; f: %.4f" % (
            batch_id, temp_cost,
            a, p, r, f))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start

        print(
                "Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx + 1, epoch_cost, train_num / epoch_cost, total_loss))


        # continue
        speed, acc, p, r, f = evaluate_classification_mix(data, model, "dev",
                feature_name, feature_length, feature_ans)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish


        current_score = f
        print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        save_when_get_new_best_score_on_dev = False
        save_each_epoch = True

        if save_when_get_new_best_score_on_dev:
            if current_score > best_dev:
                if data.seg:
                    print "Exceed previous best f score:", best_dev
                else:
                    print "Exceed previous best acc score:", best_dev
                if data.save_model:
                    model_name = data.model_dir + data.state_training_name + '.'\
                                 + feature_name + 'len%d'%(feature_length) + '.ans'\
                                 + '-'.join([str(_) for _ in feature_ans]) + '.score' + str(current_score)[2:-1]
                    print "Save current best model in file:", model_name
                    torch.save(model.state_dict(), model_name)
                best_dev = current_score
                best_dev_epoch = idx

        if save_each_epoch:
            if current_score > best_dev:
                best_dev = current_score
                best_dev_epoch = idx
            if data.seg:
                print "Exceed previous best f score:", best_dev
            else:
                print "Exceed previous best acc score:", best_dev
            if data.save_model:
                model_name = data.model_dir + data.state_training_name + '.'\
                             + feature_name + 'len%d'%(feature_length) + '.ans'\
                             + '-'.join([str(_) for _ in feature_ans]) + '.score' + str(current_score)[2:-1]
                print "Save model at %d epoch in file:" % (idx), model_name
                torch.save(model.state_dict(), model_name)

        ## decode test
        speed, acc, p, r, f = evaluate_classification_mix(data, model, "test",
            feature_name, feature_length, feature_ans)

        if f > best_test:
            best_test = f
            best_test_epoch = idx

        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        print('best_dev_score: %.4f, best_dev_epoch:%d'%(best_dev,best_dev_epoch))
        print('best_test_score: %.4f, best_test_epoch:%d'%(best_test,best_test_epoch))
        gc.collect()

def load_classification_model_decode(data, name, topN=5, threshold='0.0'):
    print "Load Models from dir: ", data.model_dir
    input_files = os.listdir(data.model_dir)
    print 'find %d files.'% len(input_files)
    ans_dict = {}
    mdic = {}
    for input_file in input_files:
        m_name = input_file[::-1].split('.',1)[1][::-1]
        if m_name in mdic:
            mdic[m_name].append(input_file)
        else:
            mdic[m_name] = []
            mdic[m_name].append(input_file)

    for k, v in mdic.items():
        print('**** in config: %s ****' % k)
        fe_old = 0
        temp_start = time.time()
        ans_lists = []
        ans_lists_test_score = []
        theLen = len(k) + 17
        # vnew = []
        # for vt in v:
        #     if len(vt) != theLen:
        #         print ('drop out: %s' % vt)
        #     else:
        #         vnew.append()
        # v = vnew
        v = sorted(v, reverse=True)

        for i in range(len(v)):
            if topN and i >= topN:
                continue
            if v[i] < k+'.score'+ str(threshold)[2:]:
                continue
            print('\t checkpiont: %s' % v[i])
            name_split = k.split('.')
            feature_name = name_split[2].split('len')[0]
            feature_length = int(name_split[2].split('len')[1])
            class_num = data.class_num
            feature_ans = [int(_) for _ in name_split[3][3:].split('-')]

            feature_name_id = data.substring_names.index(feature_name)
            model = LSTMText(data, feature_name, feature_ans, feature_length)
            if data.gpu:
                model.cuda()
            model.load_state_dict(torch.load(os.path.join(data.model_dir, v[i])))

            # speed, acc, p, r, f = evaluate_classification(data, model, "test",
            #     feature_name, feature_length, feature_ans)

            feature_name_id = data.substring_names.index(feature_name)
            class_num = 2

            if name == "train":
                instances = data.substring_train_Ids[feature_name_id][feature_length]
            elif name == "dev":
                instances = data.substring_dev_Ids[feature_name_id][feature_length]
            elif name == 'test':
                instances = data.substring_test_Ids[feature_name_id][feature_length]
            else:
                print "Error: wrong evaluate name,", name

            right_token = 0
            whole_token = 0

            nbest_pred_results = []
            pred_scores = []

            pred_results = []  # total pred result
            gold_results = []  # total gold result

            # set model in eval model
            model.eval()
            batch_size = data.batch_size
            start_time = time.time()
            train_num = len(instances)
            total_batch = train_num // batch_size + 1

            pt = 0
            nt = 0
            pf = 0
            nf = 0
            pte = 0
            nte = 0
            pfe = 0
            nfe = 0
            b = 0.00000001

            ans_list = []
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance = instances[start:end]
                if not instance:
                    continue


                batch_word, batch_wordlen, batch_label, mask = \
                    batchify_with_label_norank_classification(instance, class_num, feature_ans, data.gpu)
                score = model(batch_word, batch_wordlen, mask)

                def get_batch_ensemble_score(ans_lists, b_i, score):
                    ensemble_score = []
                    class_num = len(score[0])

                    for i in range(len(score)):
                        score_item = [score[i][_].tolist() for _ in range(class_num)]
                        for j in range(len(ans_lists)):

                            for k in range(class_num):
                                score_item[k] += ans_lists[j][0][b_i + i][k]
                        ensemble_score.append(score_item)
                    return ensemble_score

                ensemble_score = get_batch_ensemble_score(ans_lists, batch_id*batch_size, score)

                for i in range(len(score)):
                    ans_list.append(score[i].tolist())


                predict = score.data.topk(1, dim=1)[1].cpu().tolist()
                ensemble_predict = max(ensemble_score)

                ensemble_predict = [[_.index(max(_))] for _ in ensemble_score]
                true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

                predict_label_and_marked_label_list = []
                ensemble_predict_label_and_marked_label_list = []
                for jj in range(batch_label.size(0)):
                    if jj < end - start:
                        predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))
                        ensemble_predict_label_and_marked_label_list.append((ensemble_predict[jj], true_target[jj]))

                for pre, tru in predict_label_and_marked_label_list:
                    if tru[0] == 0:
                        if pre[0] == 0:
                            nf += 1
                        else:
                            pf += 1
                    else:
                        if pre[0] == 0:
                            nt += 1
                        else:
                            pt += 1

                for pre, tru in ensemble_predict_label_and_marked_label_list:
                    if tru[0] == 0:
                        if pre[0] == 0:
                            nfe += 1
                        else:
                            pfe += 1
                    else:
                        if pre[0] == 0:
                            nte += 1
                        else:
                            pte += 1



            p = pt / (pt + pf + b)
            r = pt / (pt + nt + b)
            a = (pt + nf) / (len(instances) + b)
            f = 2 * p * r / (p + r + b)

            pe = pte / (pte + pfe + b)
            re = pte / (pte + nte + b)
            ae = (pte + nfe) / (len(instances) + b)
            fe = 2 * pe * re / (pe + re + b)

            temp_time = time.time()
            temp_cost = temp_time - temp_start
            speed = train_num / temp_cost

            print("\tTest: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                  "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, a, p, r, f))
            print("\tEnsemble Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                  "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, ae, pe, re, fe))

            no_threshold_of_p = False
            if no_threshold_of_p or p > 0.5:
                ans_lists.append([ans_list, p])
            else:
                print('not save via p under 0.5')

            break_when_fe_not_raise = True
            if break_when_fe_not_raise and fe < fe_old:
                print('break_when_fe_not_raise')
                break

            fe_old = fe

        if ans_lists != []:
            ans_dict[k] = ans_lists

        gc.collect()

    f1 = file(name + '.ans_dict.pkl', 'wb')
    pickle.dump(ans_dict, f1, True)
    f1.close()
    return ans_dict

def classification_load_selected_models_and_decode(data, name, topN=1, rankin=['p','r','f']):
    # predict from whole dataset not substring dataset to fix the problem !!! 3
    # outfiledir = 'save-substring/'

    recordname_file_path = os.path.join(data.middle_dir, data.task_name)
    recordname_file_name = name + '.' + data.feature_name + '.score_dict.pkl'
    recordname = os.path.join(recordname_file_path, recordname_file_name)
    # recordname = 'record.pkl'

    ans_dict = {}

    print "Load Models from dir: ", recordname_file_path
    print "rankin: ", rankin
    with open(recordname, 'rb') as fpkl:
        classifer_count_dic = pickle.load(fpkl)

    for k, v in classifer_count_dic.items():
        print '*'*10, k
        feature_name = k[0]
        feature_length = k[1]
        feature_ans = list(k[2])
        feature_name_id = data.substring_names.index(feature_name)
        # feature_ans = [int(_) for _ in name_split[3][3:].split('-')]
        ans_lists = []
        temp_start = time.time()

        class_num = 2
        instances_raw = []
        if name == "train":
            instances = data.substring_train_Ids[feature_name_id][feature_length]
            for i in range(len(data.train_Ids)):
                for j in range(len(data.train_Ids[i][0]) - feature_length + 1):
                    instances_raw.append([data.train_Ids[i][0][j:j + feature_length]])
        elif name == "dev":
            instances = data.substring_dev_Ids[feature_name_id][feature_length]
            for i in range(len(data.dev_Ids)):
                for j in range(len(data.dev_Ids[i][0]) - feature_length + 1):
                    instances_raw.append([data.dev_Ids[i][0][j:j + feature_length]])
        elif name == 'test':
            instances = data.substring_test_Ids[feature_name_id][feature_length]
            for i in range(len(data.test_Ids)):
                for j in range(len(data.test_Ids[i][0]) - feature_length + 1):
                    instances_raw.append([data.test_Ids[i][0][j:j + feature_length]])
        else:
            print "Error: wrong evaluate name,", name

        def get_ans_list():
            fe_old = 0
            ans_list = []
            ## ensemble is
            use_ensemble = False
            ensemble_lists = []
            for vi in range(topN):
                print '*'*10, v[vi]
                model = LSTMText(data, feature_name, feature_ans, feature_length)
                if data.gpu:
                    model.cuda()
                model.load_state_dict(torch.load(os.path.join(data.model_dir, v[vi][0])))

                right_token = 0
                whole_token = 0

                nbest_pred_results = []
                pred_scores = []

                pred_results = []  # total pred result
                gold_results = []  # total gold result

                # set model in eval model
                model.eval()
                batch_size = data.batch_size
                start_time = time.time()
                train_num = len(instances)
                total_batch = train_num // batch_size + 1
                train_num_raw = len(instances_raw)
                total_batch_raw = train_num_raw // batch_size + 1


                pt = 0
                nt = 0
                pf = 0
                nf = 0
                pte = 0
                nte = 0
                pfe = 0
                nfe = 0
                b = 0.00000001
                for batch_id in range(total_batch):

                    start = batch_id * batch_size
                    end = (batch_id + 1) * batch_size
                    if end > train_num:
                        end = train_num
                    instance = instances[start:end]
                    if not instance:
                        continue

                    batch_word, batch_wordlen, batch_label, mask = \
                        batchify_with_label_norank_classification(instance, class_num, feature_ans, data.gpu)
                    score = model(batch_word, batch_wordlen, mask)

                    def get_batch_ensemble_score(ans_lists, b_i, score):
                        ensemble_score = []
                        class_num = len(score[0])

                        for i in range(len(score)):
                            score_item = [score[i][_].tolist() for _ in range(class_num)]
                            for j in range(len(ans_lists)):

                                for k in range(class_num):
                                    score_item[k] += ans_lists[j][0][b_i + i][k]
                            ensemble_score.append(score_item)
                        return ensemble_score

                    if use_ensemble: ensemble_score = get_batch_ensemble_score(ensemble_lists, batch_id * batch_size, score)

                    predict = score.data.topk(1, dim=1)[1].cpu().tolist()
                    if use_ensemble: ensemble_predict = max(ensemble_score)

                    if use_ensemble: ensemble_predict = [[_.index(max(_))] for _ in ensemble_score]
                    true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

                    predict_label_and_marked_label_list = []
                    if use_ensemble: ensemble_predict_label_and_marked_label_list = []
                    for jj in range(batch_label.size(0)):
                        if jj < end - start:
                            predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))
                            if use_ensemble: ensemble_predict_label_and_marked_label_list.append((ensemble_predict[jj], true_target[jj]))

                    for pre, tru in predict_label_and_marked_label_list:
                        if tru[0] == 0:
                            if pre[0] == 0:
                                nf += 1
                            else:
                                pf += 1
                        else:
                            if pre[0] == 0:
                                nt += 1
                            else:
                                pt += 1
                    if use_ensemble:
                        for pre, tru in ensemble_predict_label_and_marked_label_list:
                            if tru[0] == 0:
                                if pre[0] == 0:
                                    nfe += 1
                                else:
                                    pfe += 1
                            else:
                                if pre[0] == 0:
                                    nte += 1
                                else:
                                    pte += 1

                for batch_id in range(total_batch_raw):
                    start = batch_id * batch_size
                    end = (batch_id + 1) * batch_size
                    if end > train_num:
                        end = train_num
                    instance_raw = instances_raw[start:end]
                    if not instance_raw:
                        continue

                    batch_word, batch_wordlen, mask = \
                        batchify_without_label_norank_classification(instance_raw, data.gpu)
                    score_raw = model(batch_word, batch_wordlen, mask)

                    for i in range(len(score_raw)):
                        ans_list.append(score_raw[i].tolist())

                p = pt / (pt + pf + b)
                r = pt / (pt + nt + b)
                a = (pt + nf) / (len(instances) + b)
                f = 2 * p * r / (p + r + b)

                pe = pte / (pte + pfe + b)
                re = pte / (pte + nte + b)
                ae = (pte + nfe) / (len(instances) + b)
                fe = 2 * pe * re / (pe + re + b)

                temp_time = time.time()
                temp_cost = temp_time - temp_start
                speed = train_num / temp_cost

                print("\tTest: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                      "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, a, p, r, f))
                print("\tEnsemble Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                      "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, ae, pe, re, fe))

                break_when_fe_not_raise = True
                if break_when_fe_not_raise and fe < fe_old:
                    print('break_when_fe_not_raise')
                    break

                fe_old = fe
                print('get ans of {}'.format(v[vi]))
            return ans_list, p, r, f

        if 'p' in rankin:
            v = sorted(v, key=lambda k: k[1], reverse=True)
            ans_list, p, r, f = get_ans_list()
            ans_lists.append([ans_list, p, r, f])
        if 'r' in rankin:
            v = sorted(v, key=lambda k: k[2], reverse=True)
            ans_list, p, r, f = get_ans_list()
            ans_lists.append([ans_list, p, r, f])
        if 'f' in rankin:
            v = sorted(v, key=lambda k: k[3], reverse=True)
            ans_list, p, r, f = get_ans_list()
            ans_lists.append([ans_list, p, r, f])
        ans_list, p, r, f = get_ans_list()




        if ans_lists != []:
            ans_dict[k] = ans_lists

        gc.collect()

    save_file_name = name + '.' + data.feature_name + '.ans_dict.pkl'
    save_file_path = os.path.join(data.middle_dir, data.task_name)
    save_file_path = os.path.join(save_file_path, save_file_name)
    f1 = file(save_file_path, 'wb')
    pickle.dump(ans_dict, f1, True)
    f1.close()
    print 'done'
    return ans_dict

def classification_get_ans_list(data, name, topN=1, rankin=['p','r','f']):
    # name is classification_load_selected_models_and_decode_v2

    # predict from whole dataset not substring dataset to fix the problem !!! 3
    # outfiledir = 'save-substring/'

    recordname_file_path = os.path.join(data.middle_dir, data.task_name)
    recordname_file_name = name + '.' + data.feature_name + '.score_dict.pkl'
    recordname = os.path.join(recordname_file_path, recordname_file_name)
    # recordname = 'record.pkl'

    ans_dict = {}

    print "Load Models from dir: ", recordname_file_path
    print "rankin: ", rankin
    with open(recordname, 'rb') as fpkl:
        classifer_count_dic = pickle.load(fpkl)

    for k, v in classifer_count_dic.items():
        print '*' * 10, k
        feature_name = k[0]
        feature_length = k[1]
        feature_ans = list(k[2])
        feature_name_id = data.substring_names.index(feature_name)
        # feature_ans = [int(_) for _ in name_split[3][3:].split('-')]
        ans_lists = []
        temp_start = time.time()

        class_num = 2
        instances_raw = []
        if name == "train":
            instances = data.substring_train_Ids[feature_name_id][feature_length]
            for i in range(len(data.train_Ids)):
                for j in range(len(data.train_Ids[i][0]) - feature_length + 1):
                    instances_raw.append([data.train_Ids[i][0][j:j + feature_length]])
        elif name == "dev":
            instances = data.substring_dev_Ids[feature_name_id][feature_length]
            for i in range(len(data.dev_Ids)):
                for j in range(len(data.dev_Ids[i][0]) - feature_length + 1):
                    instances_raw.append([data.dev_Ids[i][0][j:j + feature_length]])
        elif name == 'test':
            instances = data.substring_test_Ids[feature_name_id][feature_length]
            for i in range(len(data.test_Ids)):
                for j in range(len(data.test_Ids[i][0]) - feature_length + 1):
                    instances_raw.append([data.test_Ids[i][0][j:j + feature_length]])
        else:
            print "Error: wrong evaluate name,", name

        def get_ans_list():
            fe_old = 0
            ans_list = []
            ## ensemble is
            use_ensemble = False
            ensemble_lists = []
            for vi in range(topN):
                print '*'*10, v[vi]
                model = LSTMText(data, feature_name, feature_ans, feature_length)
                if data.gpu:
                    model.cuda()
                model.load_state_dict(torch.load(os.path.join(data.model_dir, v[vi][0])))

                right_token = 0
                whole_token = 0

                nbest_pred_results = []
                pred_scores = []

                pred_results = []  # total pred result
                gold_results = []  # total gold result

                # set model in eval model
                model.eval()
                batch_size = data.batch_size
                start_time = time.time()
                train_num = len(instances)
                total_batch = train_num // batch_size + 1
                train_num_raw = len(instances_raw)
                total_batch_raw = train_num_raw // batch_size + 1


                pt = 0
                nt = 0
                pf = 0
                nf = 0
                pte = 0
                nte = 0
                pfe = 0
                nfe = 0
                b = 0.00000001
                for batch_id in range(total_batch):

                    start = batch_id * batch_size
                    end = (batch_id + 1) * batch_size
                    if end > train_num:
                        end = train_num
                    instance = instances[start:end]
                    if not instance:
                        continue

                    batch_word, batch_wordlen, batch_label, mask = \
                        batchify_with_label_norank_classification(instance, class_num, feature_ans, data.gpu)
                    score = model(batch_word, batch_wordlen, mask)

                    def get_batch_ensemble_score(ans_lists, b_i, score):
                        ensemble_score = []
                        class_num = len(score[0])

                        for i in range(len(score)):
                            score_item = [score[i][_].tolist() for _ in range(class_num)]
                            for j in range(len(ans_lists)):

                                for k in range(class_num):
                                    score_item[k] += ans_lists[j][0][b_i + i][k]
                            ensemble_score.append(score_item)
                        return ensemble_score

                    if use_ensemble: ensemble_score = get_batch_ensemble_score(ensemble_lists, batch_id * batch_size, score)

                    predict = score.data.topk(1, dim=1)[1].cpu().tolist()
                    if use_ensemble: ensemble_predict = max(ensemble_score)

                    if use_ensemble: ensemble_predict = [[_.index(max(_))] for _ in ensemble_score]
                    true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

                    predict_label_and_marked_label_list = []
                    if use_ensemble: ensemble_predict_label_and_marked_label_list = []
                    for jj in range(batch_label.size(0)):
                        if jj < end - start:
                            predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))
                            if use_ensemble: ensemble_predict_label_and_marked_label_list.append((ensemble_predict[jj], true_target[jj]))

                    for pre, tru in predict_label_and_marked_label_list:
                        if tru[0] == 0:
                            if pre[0] == 0:
                                nf += 1
                            else:
                                pf += 1
                        else:
                            if pre[0] == 0:
                                nt += 1
                            else:
                                pt += 1
                    if use_ensemble:
                        for pre, tru in ensemble_predict_label_and_marked_label_list:
                            if tru[0] == 0:
                                if pre[0] == 0:
                                    nfe += 1
                                else:
                                    pfe += 1
                            else:
                                if pre[0] == 0:
                                    nte += 1
                                else:
                                    pte += 1

                for batch_id in range(total_batch_raw):
                    start = batch_id * batch_size
                    end = (batch_id + 1) * batch_size
                    if end > train_num:
                        end = train_num
                    instance_raw = instances_raw[start:end]
                    if not instance_raw:
                        continue

                    batch_word, batch_wordlen, mask = \
                        batchify_without_label_norank_classification(instance_raw, data.gpu)
                    score_raw = model(batch_word, batch_wordlen, mask)

                    for i in range(len(score_raw)):
                        ans_list.append(score_raw[i].tolist())

                p = pt / (pt + pf + b)
                r = pt / (pt + nt + b)
                a = (pt + nf) / (len(instances) + b)
                f = 2 * p * r / (p + r + b)

                pe = pte / (pte + pfe + b)
                re = pte / (pte + nte + b)
                ae = (pte + nfe) / (len(instances) + b)
                fe = 2 * pe * re / (pe + re + b)

                temp_time = time.time()
                temp_cost = temp_time - temp_start
                speed = train_num / temp_cost

                print("\tTest: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                      "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, a, p, r, f))
                print("\tEnsemble Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                      "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, ae, pe, re, fe))

                break_when_fe_not_raise = True
                if break_when_fe_not_raise and fe < fe_old:
                    print('break_when_fe_not_raise')
                    break

                fe_old = fe
                print('get ans of {}'.format(v[vi]))
            return ans_list, p, r, f

        if 'p' in rankin:
            v = sorted(v, key=lambda k: k[1], reverse=True)
            ans_list, p, r, f = get_ans_list()
            ans_lists.append([ans_list, p, r, f])
        if 'r' in rankin:
            v = sorted(v, key=lambda k: k[2], reverse=True)
            ans_list, p, r, f = get_ans_list()
            ans_lists.append([ans_list, p, r, f])
        if 'f' in rankin:
            v = sorted(v, key=lambda k: k[3], reverse=True)
            ans_list, p, r, f = get_ans_list()
            ans_lists.append([ans_list, p, r, f])
        ans_list, p, r, f = get_ans_list()




        if ans_lists != []:
            ans_dict[k] = ans_lists

        gc.collect()

    save_file_name = name + '.' + data.feature_name + '.ans_dict.pkl'
    save_file_path = os.path.join(data.middle_dir, data.task_name)
    save_file_path = os.path.join(save_file_path, save_file_name)
    f1 = file(save_file_path, 'wb')

    pickle.dump(ans_dict, f1, True)
    f1.close()
    print 'done'
    return ans_dict

def classification_get_ans_list_mix(data, name, topN=1, rankin=['p','r','f']):
    # name is classification_load_selected_models_and_decode_v2

    # predict from whole dataset not substring dataset to fix the problem !!! 3
    # outfiledir = 'save-substring/'

    recordname_file_path = os.path.join(data.middle_dir, data.task_name)
    recordname_file_name = name + '.' + data.feature_name + '-selected' + '.score_dict.pkl'
    recordname = os.path.join(recordname_file_path, recordname_file_name)
    # recordname = 'record.pkl'

    ans_dict = {}

    print "Load Models from dir: ", recordname_file_path
    print "rankin: ", rankin
    with open(recordname, 'rb') as fpkl:
        classifer_count_dic = pickle.load(fpkl)

    for k, v in classifer_count_dic.items():
        print '*'*10, k
        feature_name = k[0]
        feature_length = k[1]
        feature_ans = list(k[2])
        feature_name_id_word = data.substring_names.index('word')
        feature_name_id_pos = data.substring_names.index('pos')
        # feature_ans = [int(_) for _ in name_split[3][3:].split('-')]
        ans_lists = []
        temp_start = time.time()




        class_num = 2
        instances_raw = []
        instances_raw_word = []
        instances_raw_pos = []
        if name == "train":
            indata_word = data.substring_train_Ids[feature_name_id_word][feature_length]
            indata_pos = data.substring_train_Ids[feature_name_id_pos][feature_length]
            instances = [[[[indata_word[indata_idx][0][len_idx], indata_pos[indata_idx][0][len_idx]]
                           for len_idx in range(len(indata_word[indata_idx][0]))],
                          indata_word[indata_idx][1]]
                         for indata_idx in range(len(indata_word))]
            for i in range(len(data.train_Ids)):
                for j in range(len(data.train_Ids[i][0]) - feature_length + 1):
                    instances_raw_word.append([data.train_Ids[i][0][j:j + feature_length]])
                    instances_raw_pos.append([[irp[0] for irp in [data.train_Ids[i][1][j:j + feature_length]][0]]])
            instances_raw = [[[[instances_raw_word[indata_idx][0][len_idx], instances_raw_pos[indata_idx][0][len_idx]]
                               for len_idx in range(len(instances_raw_word[indata_idx][0]))]]
                             for indata_idx in range(len(instances_raw_word))]
        elif name == "dev":
            indata_word = data.substring_dev_Ids[feature_name_id_word][feature_length]
            indata_pos = data.substring_dev_Ids[feature_name_id_pos][feature_length]
            instances = [[[[indata_word[indata_idx][0][len_idx], indata_pos[indata_idx][0][len_idx]]
                           for len_idx in range(len(indata_word[indata_idx][0]))],
                          indata_word[indata_idx][1]]
                         for indata_idx in range(len(indata_word))]
            for i in range(len(data.dev_Ids)):
                for j in range(len(data.dev_Ids[i][0]) - feature_length + 1):
                    instances_raw_word.append([data.dev_Ids[i][0][j:j + feature_length]])
                    instances_raw_pos.append([[irp[0] for irp in [data.dev_Ids[i][1][j:j + feature_length]][0]]])
            instances_raw = [[[[instances_raw_word[indata_idx][0][len_idx], instances_raw_pos[indata_idx][0][len_idx]]
                               for len_idx in range(len(instances_raw_word[indata_idx][0]))]]
                             for indata_idx in range(len(instances_raw_word))]
        elif name == 'test':
            indata_word = data.substring_test_Ids[feature_name_id_word][feature_length]
            indata_pos = data.substring_test_Ids[feature_name_id_pos][feature_length]
            instances = [[[[indata_word[indata_idx][0][len_idx], indata_pos[indata_idx][0][len_idx]]
                           for len_idx in range(len(indata_word[indata_idx][0]))],
                          indata_word[indata_idx][1]]
                         for indata_idx in range(len(indata_word))]
            for i in range(len(data.test_Ids)):
                for j in range(len(data.test_Ids[i][0]) - feature_length + 1):
                    instances_raw_word.append([data.test_Ids[i][0][j:j + feature_length]])
                    instances_raw_pos.append([[irp[0] for irp in [data.test_Ids[i][1][j:j + feature_length]][0]]])
            instances_raw = [[[[instances_raw_word[indata_idx][0][len_idx], instances_raw_pos[indata_idx][0][len_idx]]
                               for len_idx in range(len(instances_raw_word[indata_idx][0]))]]
                             for indata_idx in range(len(instances_raw_word))]
        else:
            print "Error: wrong evaluate name,", name

        def get_ans_list():
            fe_old = 0
            ans_list = []
            ## ensemble is
            use_ensemble = False
            ensemble_lists = []
            for vi in range(topN):
                print '*'*10, v[vi]
                model = LSTMText_mix(data, feature_name, feature_ans, feature_length)
                if data.gpu:
                    model.cuda()
                model.load_state_dict(torch.load(os.path.join(data.model_dir, v[vi][0])))

                right_token = 0
                whole_token = 0

                nbest_pred_results = []
                pred_scores = []

                pred_results = []  # total pred result
                gold_results = []  # total gold result

                # set model in eval model
                model.eval()
                batch_size = data.batch_size
                start_time = time.time()
                train_num = len(instances)
                total_batch = train_num // batch_size + 1
                train_num_raw = len(instances_raw)
                total_batch_raw = train_num_raw // batch_size + 1


                pt = 0
                nt = 0
                pf = 0
                nf = 0
                pte = 0
                nte = 0
                pfe = 0
                nfe = 0
                b = 0.00000001
                for batch_id in range(total_batch):

                    start = batch_id * batch_size
                    end = (batch_id + 1) * batch_size
                    if end > train_num:
                        end = train_num
                    instance = instances[start:end]
                    if not instance:
                        continue

                    batch_word, batch_wordlen, batch_label, mask = \
                        batchify_with_label_norank_classification(instance, class_num, feature_ans, data.gpu)
                    score = model(batch_word, batch_wordlen, mask)

                    def get_batch_ensemble_score(ans_lists, b_i, score):
                        ensemble_score = []
                        class_num = len(score[0])

                        for i in range(len(score)):
                            score_item = [score[i][_].tolist() for _ in range(class_num)]
                            for j in range(len(ans_lists)):

                                for k in range(class_num):
                                    score_item[k] += ans_lists[j][0][b_i + i][k]
                            ensemble_score.append(score_item)
                        return ensemble_score

                    if use_ensemble: ensemble_score = get_batch_ensemble_score(ensemble_lists, batch_id * batch_size, score)

                    predict = score.data.topk(1, dim=1)[1].cpu().tolist()
                    if use_ensemble: ensemble_predict = max(ensemble_score)

                    if use_ensemble: ensemble_predict = [[_.index(max(_))] for _ in ensemble_score]
                    true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

                    predict_label_and_marked_label_list = []
                    if use_ensemble: ensemble_predict_label_and_marked_label_list = []
                    for jj in range(batch_label.size(0)):
                        if jj < end - start:
                            predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))
                            if use_ensemble: ensemble_predict_label_and_marked_label_list.append((ensemble_predict[jj], true_target[jj]))

                    for pre, tru in predict_label_and_marked_label_list:
                        if tru[0] == 0:
                            if pre[0] == 0:
                                nf += 1
                            else:
                                pf += 1
                        else:
                            if pre[0] == 0:
                                nt += 1
                            else:
                                pt += 1
                    if use_ensemble:
                        for pre, tru in ensemble_predict_label_and_marked_label_list:
                            if tru[0] == 0:
                                if pre[0] == 0:
                                    nfe += 1
                                else:
                                    pfe += 1
                            else:
                                if pre[0] == 0:
                                    nte += 1
                                else:
                                    pte += 1

                for batch_id in range(total_batch_raw):
                    start = batch_id * batch_size
                    end = (batch_id + 1) * batch_size
                    if end > train_num:
                        end = train_num
                    instance_raw = instances_raw[start:end]
                    if not instance_raw:
                        continue

                    batch_word, batch_wordlen, mask = \
                        batchify_without_label_norank_classification(instance_raw, data.gpu)
                    score_raw = model(batch_word, batch_wordlen, mask)

                    for i in range(len(score_raw)):
                        ans_list.append(score_raw[i].tolist())

                p = pt / (pt + pf + b)
                r = pt / (pt + nt + b)
                a = (pt + nf) / (len(instances) + b)
                f = 2 * p * r / (p + r + b)

                pe = pte / (pte + pfe + b)
                re = pte / (pte + nte + b)
                ae = (pte + nfe) / (len(instances) + b)
                fe = 2 * pe * re / (pe + re + b)

                temp_time = time.time()
                temp_cost = temp_time - temp_start
                speed = train_num / temp_cost

                print("\tTest: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                      "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, a, p, r, f))
                print("\tEnsemble Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                      "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, ae, pe, re, fe))

                break_when_fe_not_raise = True
                if break_when_fe_not_raise and fe < fe_old:
                    print('break_when_fe_not_raise')
                    break

                fe_old = fe
                print('get ans of {}'.format(v[vi]))
            return ans_list, p, r, f

        if 'p' in rankin:
            v = sorted(v, key=lambda k: k[1], reverse=True)
            ans_list, p, r, f = get_ans_list()
            ans_lists.append([ans_list, p, r, f])
        if 'r' in rankin:
            v = sorted(v, key=lambda k: k[2], reverse=True)
            ans_list, p, r, f = get_ans_list()
            ans_lists.append([ans_list, p, r, f])
        if 'f' in rankin:
            v = sorted(v, key=lambda k: k[3], reverse=True)
            ans_list, p, r, f = get_ans_list()
            ans_lists.append([ans_list, p, r, f])
        ans_list, p, r, f = get_ans_list()




        if ans_lists != []:
            ans_dict[k] = ans_lists

        gc.collect()

    save_file_name = name + '.' + data.feature_name + '.ans_dict.pkl'
    save_file_path = os.path.join(data.middle_dir, data.task_name)
    save_file_path = os.path.join(save_file_path, save_file_name)
    f1 = file(save_file_path, 'wb')
    pickle.dump(ans_dict, f1, True)
    f1.close()
    print 'done'
    return ans_dict

def load_classification_model_decode_ver2(data, name, topN=5, threshold='0.0'):
    # predict from whole dataset not substring dataset to fix the problem !!! 3
    print "Load Models from dir: ", data.model_dir
    input_files = os.listdir(data.model_dir)
    print 'find %d files.'% len(input_files)
    save_name = 'class.' + name + '.ans_dict.pkl'
    ans_dict = {}
    mdic = {}
    for input_file in input_files:
        m_name = input_file[::-1].split('.',1)[1][::-1]
        if m_name in mdic:
            mdic[m_name].append(input_file)
        else:
            mdic[m_name] = []
            mdic[m_name].append(input_file)

    for k, v in mdic.items():
        print('**** in config: %s ****' % k)
        fe_old = 0
        temp_start = time.time()
        ans_lists = []
        v = sorted(v, reverse=True)

        for i in range(len(v)):
            if topN and i >= topN:
                continue
            if v[i] < k+'.score'+ str(threshold)[2:]:
                continue
            print('\t checkpiont: %s' % v[i])
            name_split = k.split('.')
            feature_name = name_split[2].split('len')[0]
            feature_length = int(name_split[2].split('len')[1])
            class_num = data.class_num
            feature_ans = [int(_) for _ in name_split[3][3:].split('-')]

            model = LSTMText(data, feature_name, feature_ans, feature_length)
            if data.gpu:
                model.cuda()
            model.load_state_dict(torch.load(os.path.join(data.model_dir, v[i])))


            feature_name_id = data.substring_names.index(feature_name)
            class_num = 2

            instances_raw = []

            if name == "train":
                instances = data.substring_train_Ids[feature_name_id][feature_length]
                for i in range(len(data.train_Ids)):
                    for j in range(len(data.train_Ids[i][0]) - feature_length + 1):
                        instances_raw.append([data.train_Ids[i][0][j:j+feature_length]])
            elif name == "dev":
                instances = data.substring_dev_Ids[feature_name_id][feature_length]
                for i in range(len(data.dev_Ids)):
                    for j in range(len(data.dev_Ids[i][0]) - feature_length + 1):
                        instances_raw.append([data.dev_Ids[i][0][j:j+feature_length]])
            elif name == 'test':
                instances = data.substring_test_Ids[feature_name_id][feature_length]
                for i in range(len(data.test_Ids)):
                    for j in range(len(data.test_Ids[i][0]) - feature_length + 1):
                        instances_raw.append([data.test_Ids[i][0][j:j+feature_length]])
            else:
                print "Error: wrong evaluate name,", name

            right_token = 0
            whole_token = 0

            nbest_pred_results = []
            pred_scores = []

            pred_results = []  # total pred result
            gold_results = []  # total gold result

            # set model in eval model
            model.eval()
            batch_size = data.batch_size
            start_time = time.time()
            train_num = len(instances)
            total_batch = train_num // batch_size + 1
            train_num_raw = len(instances_raw)
            total_batch_raw = train_num_raw // batch_size + 1

            pt = 0
            nt = 0
            pf = 0
            nf = 0
            pte = 0
            nte = 0
            pfe = 0
            nfe = 0
            b = 0.00000001

            ans_list = []
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance = instances[start:end]
                if not instance:
                    continue

                batch_word, batch_wordlen, batch_label, mask = \
                    batchify_with_label_norank_classification(instance, class_num, feature_ans, data.gpu)
                score = model(batch_word, batch_wordlen, mask)

                def get_batch_ensemble_score(ans_lists, b_i, score):
                    ensemble_score = []
                    class_num = len(score[0])

                    for i in range(len(score)):
                        score_item = [score[i][_].tolist() for _ in range(class_num)]
                        for j in range(len(ans_lists)):

                            for k in range(class_num):
                                score_item[k] += ans_lists[j][0][b_i + i][k]
                        ensemble_score.append(score_item)
                    return ensemble_score

                ensemble_score = get_batch_ensemble_score(ans_lists, batch_id*batch_size, score)


                predict = score.data.topk(1, dim=1)[1].cpu().tolist()
                ensemble_predict = max(ensemble_score)

                ensemble_predict = [[_.index(max(_))] for _ in ensemble_score]
                true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

                predict_label_and_marked_label_list = []
                ensemble_predict_label_and_marked_label_list = []
                for jj in range(batch_label.size(0)):
                    if jj < end - start:
                        predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))
                        ensemble_predict_label_and_marked_label_list.append((ensemble_predict[jj], true_target[jj]))

                for pre, tru in predict_label_and_marked_label_list:
                    if tru[0] == 0:
                        if pre[0] == 0:
                            nf += 1
                        else:
                            pf += 1
                    else:
                        if pre[0] == 0:
                            nt += 1
                        else:
                            pt += 1

                for pre, tru in ensemble_predict_label_and_marked_label_list:
                    if tru[0] == 0:
                        if pre[0] == 0:
                            nfe += 1
                        else:
                            pfe += 1
                    else:
                        if pre[0] == 0:
                            nte += 1
                        else:
                            pte += 1

            for batch_id in range(total_batch_raw):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance_raw = instances_raw[start:end]
                if not instance_raw:
                    continue

                batch_word, batch_wordlen, mask = \
                    batchify_without_label_norank_classification(instance_raw, data.gpu)
                score_raw = model(batch_word, batch_wordlen, mask)

                for i in range(len(score_raw)):
                    ans_list.append(score_raw[i].tolist())



            p = pt / (pt + pf + b)
            r = pt / (pt + nt + b)
            a = (pt + nf) / (len(instances) + b)
            f = 2 * p * r / (p + r + b)

            pe = pte / (pte + pfe + b)
            re = pte / (pte + nte + b)
            ae = (pte + nfe) / (len(instances) + b)
            fe = 2 * pe * re / (pe + re + b)

            temp_time = time.time()
            temp_cost = temp_time - temp_start
            speed = train_num / temp_cost

            print("\tTest: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                  "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, a, p, r, f))
            print("\tEnsemble Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                  "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, ae, pe, re, fe))

            no_threshold_of_p = False
            if no_threshold_of_p or p > 0.5:
                ans_lists.append([ans_list, p])
            else:
                print('not save via p under 0.5')

            break_when_fe_not_raise = True
            if break_when_fe_not_raise and fe < fe_old:
                print('break_when_fe_not_raise')
                break

            fe_old = fe

        if ans_lists != []:
            ans_dict[k] = ans_lists

        gc.collect()

    f1 = file(save_name, 'wb')
    pickle.dump(ans_dict, f1, True)
    f1.close()
    return ans_dict

def classification_get_scores(data, name, topN=1, threshold='0.0'):
    # output: a list of mode names which's predict recall or f1 reaches top N
    print "Load Models from dir: ", data.model_dir
    input_files = os.listdir(data.model_dir)
    print 'find %d files.' % len(input_files)
    score_dict = {}
    mdic = {}
    for input_file in input_files:
        m_name = input_file[::-1].split('.',1)[1][::-1]
        if m_name in mdic:
            mdic[m_name].append(input_file)
        else:
            mdic[m_name] = []
            mdic[m_name].append(input_file)

    proc_num_all = len(mdic)
    proc_num_now = 1
    for k, v in mdic.items():
        print('process: {}/{}'.format(proc_num_now, proc_num_all))
        proc_num_now += 1

        print('**** in config: %s ****' % k)
        fe_old = 0
        temp_start = time.time()
        ans_lists = []
        v = sorted(v, reverse=True)

        for vi in range(len(v)):
            if vi >= topN:
                continue
            print('\t checkpiont: %s' % v[vi])
            name_split = k.split('.')
            feature_name = name_split[2].split('len')[0]
            feature_length = int(name_split[2].split('len')[1])
            class_num = data.class_num
            feature_ans = [int(_) for _ in name_split[3][3:].split('-')]

            model = LSTMText(data, feature_name, feature_ans, feature_length)
            if data.gpu:
                model.cuda()
            model.load_state_dict(torch.load(os.path.join(data.model_dir, v[vi])))


            feature_name_id = data.substring_names.index(feature_name)
            class_num = 2

            instances_raw = []

            if name == "train":
                instances = data.substring_train_Ids[feature_name_id][feature_length]
                for i in range(len(data.train_Ids)):
                    for j in range(len(data.train_Ids[i][0]) - feature_length + 1):
                        instances_raw.append([data.train_Ids[i][0][j:j+feature_length]])
            elif name == "dev":
                instances = data.substring_dev_Ids[feature_name_id][feature_length]
                for i in range(len(data.dev_Ids)):
                    for j in range(len(data.dev_Ids[i][0]) - feature_length + 1):
                        instances_raw.append([data.dev_Ids[i][0][j:j+feature_length]])
            elif name == 'test':
                instances = data.substring_test_Ids[feature_name_id][feature_length]
                for i in range(len(data.test_Ids)):
                    for j in range(len(data.test_Ids[i][0]) - feature_length + 1):
                        instances_raw.append([data.test_Ids[i][0][j:j+feature_length]])
            else:
                print "Error: wrong evaluate name,", name

            model.eval()
            batch_size = data.batch_size
            start_time = time.time()
            train_num = len(instances)
            total_batch = train_num // batch_size + 1
            train_num_raw = len(instances_raw)
            total_batch_raw = train_num_raw // batch_size + 1

            pt = 0
            nt = 0
            pf = 0
            nf = 0
            pte = 0
            nte = 0
            pfe = 0
            nfe = 0
            b = 0.00000001

            ans_list = []
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance = instances[start:end]
                if not instance:
                    continue

                batch_word, batch_wordlen, batch_label, mask = \
                    batchify_with_label_norank_classification(instance, class_num, feature_ans, data.gpu)
                score = model(batch_word, batch_wordlen, mask)

                predict = score.data.topk(1, dim=1)[1].cpu().tolist()

                true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

                predict_label_and_marked_label_list = []
                for jj in range(batch_label.size(0)):
                    if jj < end - start:
                        predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))

                for pre, tru in predict_label_and_marked_label_list:
                    if tru[0] == 0:
                        if pre[0] == 0:
                            nf += 1
                        else:
                            pf += 1
                    else:
                        if pre[0] == 0:
                            nt += 1
                        else:
                            pt += 1

            for batch_id in range(total_batch_raw):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance_raw = instances_raw[start:end]
                if not instance_raw:
                    continue

                batch_word, batch_wordlen, mask = \
                    batchify_without_label_norank_classification(instance_raw, data.gpu)
                score_raw = model(batch_word, batch_wordlen, mask)

                for i in range(len(score_raw)):
                    ans_list.append(score_raw[i].tolist())



            p = pt / (pt + pf + b)
            r = pt / (pt + nt + b)
            a = (pt + nf) / (len(instances) + b)
            f = 2 * p * r / (p + r + b)

            temp_time = time.time()
            temp_cost = temp_time - temp_start
            speed = train_num / temp_cost

            print("\tTest: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                  "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, a, p, r, f))

            score_dict[v[vi]] = [a, p, r, f]
            print(score_dict[v[vi]])
            temp_start = time.time()

        gc.collect()

    file_save_path = os.path.join(data.middle_dir, data.task_name)
    file_name = name + '.' + data.feature_name + '.score_dict.pkl'
    file_save_path_name = os.path.join(file_save_path, file_name)
    f1 = file(file_save_path_name, 'wb')
    pickle.dump(score_dict, f1, True)
    f1.close()

    return score_dict

def classification_get_scores_mix(data, name, topN=15, threshold='0.0'):
    # output: a list of mode names which's predict recall or f1 reaches top N
    print "Load Models from dir: ", data.model_dir
    input_files = os.listdir(data.model_dir)
    print 'find %d files.' % len(input_files)
    score_dict = {}
    mdic = {}
    for input_file in input_files:
        m_name = input_file[::-1].split('.',1)[1][::-1]
        if m_name in mdic:
            mdic[m_name].append(input_file)
        else:
            mdic[m_name] = []
            mdic[m_name].append(input_file)

    proc_num_all = len(mdic)
    proc_num_now = 1
    for k, v in mdic.items():
        print('process: {}/{}'.format(proc_num_now, proc_num_all))
        proc_num_now += 1

        print('**** in config: %s ****' % k)
        fe_old = 0
        temp_start = time.time()
        ans_lists = []
        v = sorted(v, reverse=True)
        print('k:{},v:{}'.format(k, v))
        for vi in range(len(v)):
            if vi >= topN:
                continue
            print('\t checkpiont: %s' % v[vi])
            name_split = k.split('.')
            feature_name = name_split[2].split('len')[0]
            feature_length = int(name_split[2].split('len')[1])
            class_num = data.class_num
            feature_ans = [int(_) for _ in name_split[3][3:].split('-')]

            model = LSTMText_mix(data, feature_name, feature_ans, feature_length)
            if data.gpu:
                model.cuda()
            model.load_state_dict(torch.load(os.path.join(data.model_dir, v[vi])))


            # feature_name_id = data.substring_names.index(feature_name)
            feature_name_id_word = data.substring_names.index('word')
            feature_name_id_pos = data.substring_names.index('pos')
            class_num = 2

            instances_raw = []
            instances_raw_word = []
            instances_raw_pos = []

            if name == "train":
                # instances = data.substring_train_Ids[feature_name_id][feature_length]
                indata_word = data.substring_train_Ids[feature_name_id_word][feature_length]
                indata_pos = data.substring_train_Ids[feature_name_id_pos][feature_length]
                # indata = [[indata_word[indata_idx][0], indata_pos[indata_idx][0], indata_word[indata_idx][1]]
                #           for indata_idx in range(len(indata_word))]
                instances = [[[[indata_word[indata_idx][0][len_idx], indata_pos[indata_idx][0][len_idx]]
                               for len_idx in range(len(indata_word[indata_idx][0]))],
                              indata_word[indata_idx][1]]
                             for indata_idx in range(len(indata_word))]

                for i in range(len(data.train_Ids)):
                    for j in range(len(data.train_Ids[i][0]) - feature_length + 1):
                        instances_raw_word.append([data.train_Ids[i][0][j:j + feature_length]])
                        instances_raw_pos.append([[irp[0] for irp in [data.train_Ids[i][1][j:j + feature_length]][0]]])
                instances_raw = [
                    [[[instances_raw_word[indata_idx][0][len_idx], instances_raw_pos[indata_idx][0][len_idx]]
                      for len_idx in range(len(instances_raw_word[indata_idx][0]))]]
                    for indata_idx in range(len(instances_raw_word))]
            elif name == "dev":
                # instances = data.substring_dev_Ids[feature_name_id][feature_length]
                indata_word = data.substring_dev_Ids[feature_name_id_word][feature_length]
                indata_pos = data.substring_dev_Ids[feature_name_id_pos][feature_length]
                # indata = [[indata_word[indata_idx][0], indata_pos[indata_idx][0], indata_word[indata_idx][1]]
                #           for indata_idx in range(len(indata_word))]
                instances = [[[[indata_word[indata_idx][0][len_idx], indata_pos[indata_idx][0][len_idx]]
                               for len_idx in range(len(indata_word[indata_idx][0]))],
                              indata_word[indata_idx][1]]
                             for indata_idx in range(len(indata_word))]

                for i in range(len(data.dev_Ids)):
                    for j in range(len(data.dev_Ids[i][0]) - feature_length + 1):
                        instances_raw_word.append([data.dev_Ids[i][0][j:j + feature_length]])
                        instances_raw_pos.append([[irp[0] for irp in [data.dev_Ids[i][1][j:j + feature_length]][0]]])
                instances_raw = [
                    [[[instances_raw_word[indata_idx][0][len_idx], instances_raw_pos[indata_idx][0][len_idx]]
                      for len_idx in range(len(instances_raw_word[indata_idx][0]))]]
                    for indata_idx in range(len(instances_raw_word))]
            elif name == 'test':
                feature_name_id=0
                # instancesxx = data.substring_test_Ids[feature_name_id][feature_length]
                indata_word = data.substring_test_Ids[feature_name_id_word][feature_length]
                indata_pos = data.substring_test_Ids[feature_name_id_pos][feature_length]
                # indataxx = [[indata_word[indata_idx][0], indata_pos[indata_idx][0], indata_word[indata_idx][1]]
                #           for indata_idx in range(len(indata_word))]
                instances = [[[[indata_word[indata_idx][0][len_idx], indata_pos[indata_idx][0][len_idx]]
                            for len_idx in range(len(indata_word[indata_idx][0]))],
                           indata_word[indata_idx][1]]
                          for indata_idx in range(len(indata_word))]
                # instances_raw_xx = []
                # for i in range(len(data.test_Ids)):
                #     for j in range(len(data.test_Ids[i][0]) - feature_length + 1):
                #         instances_raw_xx.append([data.test_Ids[i][0][j:j+feature_length]])


                for i in range(len(data.test_Ids)):
                    for j in range(len(data.test_Ids[i][0]) - feature_length + 1):
                        instances_raw_word.append([data.test_Ids[i][0][j:j+feature_length]])
                        instances_raw_pos.append( [[irp[0] for irp in [data.test_Ids[i][1][j:j+feature_length]][0]]])
                instances_raw = [[[[instances_raw_word[indata_idx][0][len_idx], instances_raw_pos[indata_idx][0][len_idx]]
                               for len_idx in range(len(instances_raw_word[indata_idx][0]))]]
                             for indata_idx in range(len(instances_raw_word))]
            else:
                print "Error: wrong evaluate name,", name

            model.eval()
            batch_size = data.batch_size
            start_time = time.time()
            train_num = len(instances)
            total_batch = train_num // batch_size + 1
            train_num_raw = len(instances_raw)
            total_batch_raw = train_num_raw // batch_size + 1

            pt = 0
            nt = 0
            pf = 0
            nf = 0
            pte = 0
            nte = 0
            pfe = 0
            nfe = 0
            b = 0.00000001

            ans_list = []
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance = instances[start:end]
                if not instance:
                    continue

                batch_word, batch_wordlen, batch_label, mask = \
                    batchify_with_label_norank_classification(instance, class_num, feature_ans, data.gpu)
                score = model(batch_word, batch_wordlen, mask)

                predict = score.data.topk(1, dim=1)[1].cpu().tolist()

                true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

                predict_label_and_marked_label_list = []
                for jj in range(batch_label.size(0)):
                    if jj < end - start:
                        predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))

                for pre, tru in predict_label_and_marked_label_list:
                    if tru[0] == 0:
                        if pre[0] == 0:
                            nf += 1
                        else:
                            pf += 1
                    else:
                        if pre[0] == 0:
                            nt += 1
                        else:
                            pt += 1

            for batch_id in range(total_batch_raw):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance_raw = instances_raw[start:end]
                if not instance_raw:
                    continue

                batch_word, batch_wordlen, mask = \
                    batchify_without_label_norank_classification(instance_raw, data.gpu)
                score_raw = model(batch_word, batch_wordlen, mask)

                for i in range(len(score_raw)):
                    ans_list.append(score_raw[i].tolist())



            p = pt / (pt + pf + b)
            r = pt / (pt + nt + b)
            a = (pt + nf) / (len(instances) + b)
            f = 2 * p * r / (p + r + b)

            temp_time = time.time()
            temp_cost = temp_time - temp_start
            speed = train_num / temp_cost

            print("\tTest: time: %.2fs, speed: %.2fst/s; acc: %.4f, "
                  "p: %.4f, r: %.4f, f: %.4f" % (temp_cost, speed, a, p, r, f))

            score_dict[v[vi]] = [a, p, r, f]
            print(score_dict[v[vi]])
            temp_start = time.time()

        gc.collect()

    file_save_path = os.path.join(data.middle_dir, data.task_name)
    file_name = name + '.' + data.feature_name + '.score_dict.pkl'
    file_save_path_name = os.path.join(file_save_path, file_name)
    f1 = file(file_save_path_name, 'wb')
    pickle.dump(score_dict, f1, True)
    f1.close()

    return score_dict

def make_classification_ans_ensemble_selected_models(data, name, models='r',tagNum=6):
    # hypropare of controll
    occurrence_weight_train_ned = [0, 0.926, 0.013, 0.019, 0.029, 0.013]
    occurrence_weight_test_ned = [0, 0.934, 0.017, 0.017, 0.022, 0.008]
    use_devP_as_weight = True
    use_globel_norm = False
    pos_gather_in_middle = True
    use_occurrence_weight_pos = True
    use_occurrence_weight_neg = True


    f1 = file(name + '.ans_dict.pkl', 'rb')
    ans_dict = pickle.load(f1)
    f1.close()

    if models == 'p':
        pos_index = 0
        score_index = 1
    elif models == 'r':
        pos_index = 1
        score_index = 2
    elif models == 'f':
        pos_index = 2
        score_index = 3

    classifer_count_dic = {}
    for k, v in ans_dict.items():
        feature_length = k[1]
        feature_ans_set = k[2]
        feature_ans_highest_dev_score = v[pos_index][score_index]
        show_score = str(feature_ans_highest_dev_score)
        if feature_ans_set in classifer_count_dic:
            classifer_count_dic[feature_ans_set].append((feature_length,show_score))
        else:
            classifer_count_dic[feature_ans_set] = []
            classifer_count_dic[feature_ans_set].append((feature_length,show_score))

    print('classifier info')
    for k, v in classifer_count_dic.items():
        print('{} \t:{}'.format(k,sorted(v)))

    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    else:
        print "Error: wrong evaluate name,", name

    ans_list = []

    for i in range(len(instances)):
        ans_list_sen = [[0 for _ in range(tagNum)] for j in range(len(instances[i][0]))]
        ans_list.append(ans_list_sen)

    for k,v in ans_dict.items():
        # print('check')
        # print(ans_list[0])
        # print('**** %s ****' % k)
        # k,v = ans_dict.items()[0]
        feature_length = k[1]
        feature_ans_highest_dev_score = v[pos_index][1]

        sub_ans_list = v[pos_index]
        feature_ans = list(k[2])
        sub_ans_list_score = sub_ans_list[score_index]
        if feature_length != 1 or len(feature_ans) != 1:
            continue
        print('check')
        print(ans_list[0])
        print('**** {} ****'.format(k))
        print('\tp in dev: %f' % sub_ans_list_score)
        pos_punishment_num = feature_length * len(feature_ans)
        neg_punishment_num = feature_length * (tagNum - len(feature_ans) - 1)


        neg_punishment_rate_weight = 1
        if use_devP_as_weight:
            pos_punishment_rate = sub_ans_list_score / pos_punishment_num
            neg_punishment_rate = sub_ans_list_score / neg_punishment_num
        else:
            pos_punishment_rate = 1 / pos_punishment_num
            neg_punishment_rate = 1 / neg_punishment_num

        neg_punishment_rate *= neg_punishment_rate_weight

        sent_num = len(ans_list)
        sub_ans_list_idx = 0


        globel_mean = 1
        lenth_sub_ans_list_0 = len(sub_ans_list[0])
        if use_globel_norm:
            globel_mean = np.mean(np.abs(sub_ans_list[0])) * 2


        pos_gather_in_middle_rate = [1 for _ in range(feature_length)]
        def get_pos_gather_rate(len_l, least):
            if len_l == 1:
                return [1]
            else:
                ans_list = []
                sum = 0
                mid_len = (len_l - 1.0) / 2.0
                for i in range(len_l):
                    distance = abs(mid_len - i)
                    ans_i = 1.0 - (1.0 - least) * (distance / mid_len)
                    ans_list.append(ans_i)
                    sum += ans_i
                for i in range(len_l):
                    ans_list[i] = ans_list[i] * len_l/ sum
                return ans_list

        if pos_gather_in_middle:
            pos_gather_in_middle_rate = get_pos_gather_rate(feature_length, 0.5)

        for s in range(sent_num):
            line_iter_num = len(ans_list[s]) - feature_length + 1
            for i in range(line_iter_num):
                norm = abs(sub_ans_list[0][sub_ans_list_idx][0]) + abs(sub_ans_list[0][sub_ans_list_idx][1])
                if use_globel_norm:
                    globel_norm_rate = norm / globel_mean
                else:
                    globel_norm_rate = 1
                neg = sub_ans_list[0][sub_ans_list_idx][0] * globel_norm_rate * neg_punishment_rate / norm
                pos = sub_ans_list[0][sub_ans_list_idx][1] * globel_norm_rate * pos_punishment_rate / norm

                if use_occurrence_weight_neg:
                    not_in_ans_weights = 0
                    for ow in range(tagNum):
                        if ow not in feature_ans:
                            not_in_ans_weights += occurrence_weight_test_ned[ow]
                if use_occurrence_weight_pos:
                    in_ans_weights = 0
                    for ow in range(tagNum):
                        if ow in feature_ans:
                            in_ans_weights += occurrence_weight_test_ned[ow]
                for j in range(feature_length):
                    for ans_label in range(tagNum):
                        if ans_label in feature_ans:
                            if use_occurrence_weight_neg:
                                ans_list[s][i + j][ans_label] += \
                                    pos * (occurrence_weight_test_ned[ans_label] / in_ans_weights)
                            else:
                                ans_list[s][i + j][ans_label] += pos * pos_gather_in_middle_rate[j]
                        else:
                            if use_occurrence_weight_neg:
                                ans_list[s][i + j][ans_label] += \
                                    neg * (occurrence_weight_train_ned[ans_label] / not_in_ans_weights)
                            else:
                                ans_list[s][i + j][ans_label] += neg
                sub_ans_list_idx += 1
        print('sub_ans_list_idx is %d' % sub_ans_list_idx)
        print('sub_ans_list_idx has %d' % len(sub_ans_list[0]))


        out_predict = []
        total_word = 0
        nf = [0 for _ in range(tagNum)]
        pf = [0 for _ in range(tagNum)]
        nt = [0 for _ in range(tagNum)]
        pt = [0 for _ in range(tagNum)]
        for s in range(sent_num):
            out_predict_sent = [_.index(max(_)) for _ in ans_list[s]]
            for i in range(len(out_predict_sent)):
                if out_predict_sent[i] == 0:
                    out_predict_sent[i] = 1
            out_predict.append(out_predict_sent)

            def label_id_to_substring_label_id(label_id):
                substring_label_id = []
                for id in label_id:
                    if id == 0:
                        print('unk id in substring_label_id')
                    if id == 1:
                        substring_label_id.append(1)
                    else:
                        id_name = data.label_alphabet.instances[id-1]
                        label_id_name = id_name.split('-')[1]
                        substring_label_id.append(data.substring_label_alphabet.instance2index[label_id_name])
                return substring_label_id

            output = label_id_to_substring_label_id(instances[s][-1])
            for pre, tru in zip(out_predict[s], output):
                total_word += 1
                for k in range(tagNum):
                    if tru != k:
                        if pre != k:
                            nf[k] += 1
                        else:
                            pf[k] += 1
                    else:
                        if pre != k:
                            nt[k] += 1
                        else:
                            pt[k] += 1

        p = [0 for _ in range(tagNum)]
        r = [0 for _ in range(tagNum)]
        a = [0 for _ in range(tagNum)]
        f = [0 for _ in range(tagNum)]
        b = 0.00000001
        for k in range(tagNum):
            p[k] = pt[k] / (pt[k] + pf[k] + b)
            r[k] = pt[k] / (pt[k] + nt[k] + b)
            a[k] = (pt[k] + nf[k]) / (total_word + b)
            f[k] = 2 * p[k] * r[k] / (p[k] + r[k] + b)
            print("\t%dacc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (k, a[k], p[k], r[k], f[k]))
        print("\t*avg: %.4f, p: %.4f, r: %.4f, f: %.4f" % ((a[2]+a[3]+a[4]+a[5])/4,
            (p[2] + p[3] + p[4] + p[5])/4, (r[2] + r[3] + r[4] + r[5])/4, (f[2] + f[3] + f[4] + f[5])/4))

    return out_predict, ans_list

def make_classification_ans_ensemble(data, name, tagNum=6):
    occurrence_weight_train_ned = [0, 0.926, 0.013, 0.019, 0.029, 0.013]
    occurrence_weight_test_ned = [0, 0.934, 0.017, 0.017, 0.022, 0.008]
    f1 = file(name + '.ans_dict.pkl', 'rb')
    ans_dict = pickle.load(f1)
    f1.close()

    classifer_count_dic = {}
    for k, v in ans_dict.items():
        name_split = k.split('.')
        feature_length = int(name_split[2].split('len')[1])
        feature_ans = [int(_) for _ in name_split[3][3:].split('-')]
        feature_ans_set = tuple(sorted(feature_ans))
        feature_ans_highest_dev_score = ans_dict[k][0][1]
        show_score = str(feature_ans_highest_dev_score)[0:4]
        if feature_ans_set in classifer_count_dic:
            classifer_count_dic[feature_ans_set].append((feature_length,show_score))
        else:
            classifer_count_dic[feature_ans_set] = []
            classifer_count_dic[feature_ans_set].append((feature_length,show_score))

    print('classifier info')
    for k, v in classifer_count_dic.items():
        print('{} \t:{}'.format(k,sorted(v)))

    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    else:
        print "Error: wrong evaluate name,", name

    ans_list = []

    for i in range(len(instances)):
        ans_list_sen = [[0 for _ in range(tagNum)] for j in range(len(instances[i][0]))]
        ans_list.append(ans_list_sen)

    for k,v in ans_dict.items():
        # print('check')
        # print(ans_list[0])
        # print('**** %s ****' % k)
        # k,v = ans_dict.items()[0]
        name_split = k.split('.')
        feature_name = name_split[2].split('len')[0]
        feature_length = int(name_split[2].split('len')[1])
        class_num = data.class_num

        sub_ans_list = ans_dict[k][0]
        feature_ans = [int(_) for _ in name_split[3][3:].split('-')]
        sub_ans_list_score = sub_ans_list[1]
        # print('\tp in dev: %f' % sub_ans_list_score)
        if feature_length != 1 or len(feature_ans) != 1:
            continue
        print('check')
        print(ans_list[0])
        print('**** %s ****' % k)
        print('\tp in dev: %f' % sub_ans_list_score)
        pos_punishment_num = feature_length * len(feature_ans)
        neg_punishment_num = feature_length * (tagNum - len(feature_ans) - 1)

        use_devP_as_weight = True
        neg_punishment_rate_weight = 1
        if use_devP_as_weight:
            pos_punishment_rate = sub_ans_list_score / pos_punishment_num
            neg_punishment_rate = sub_ans_list_score / neg_punishment_num
        else:
            pos_punishment_rate = 1 / pos_punishment_num
            neg_punishment_rate = 1 / neg_punishment_num

        neg_punishment_rate *= neg_punishment_rate_weight

        sent_num = len(ans_list)
        sub_ans_list_idx = 0

        use_globel_norm = True
        globel_mean = 1
        lenth_sub_ans_list_0 = len(sub_ans_list[0])
        if use_globel_norm:
            globel_mean = np.mean(np.abs(sub_ans_list[0])) * 2

        pos_gather_in_middle = True
        pos_gather_in_middle_rate = [1 for _ in range(feature_length)]
        def get_pos_gather_rate(len_l, least):
            if len_l == 1:
                return [1]
            else:
                ans_list = []
                sum = 0
                mid_len = (len_l - 1.0) / 2.0
                for i in range(len_l):
                    distance = abs(mid_len - i)
                    ans_i = 1.0 - (1.0 - least) * (distance / mid_len)
                    ans_list.append(ans_i)
                    sum += ans_i
                for i in range(len_l):
                    ans_list[i] = ans_list[i] * len_l/ sum
                return ans_list

        if pos_gather_in_middle:
            pos_gather_in_middle_rate = get_pos_gather_rate(feature_length, 0.5)

        for s in range(sent_num):
            line_iter_num = len(ans_list[s]) - feature_length + 1
            for i in range(line_iter_num):
                norm = abs(sub_ans_list[0][sub_ans_list_idx][0]) + abs(sub_ans_list[0][sub_ans_list_idx][1])
                if use_globel_norm:
                    globel_norm_rate = norm / globel_mean
                else:
                    globel_norm_rate = 1
                neg = sub_ans_list[0][sub_ans_list_idx][0] * globel_norm_rate * neg_punishment_rate / norm
                pos = sub_ans_list[0][sub_ans_list_idx][1] * globel_norm_rate * pos_punishment_rate / norm
                use_occurrence_weight_pos = True
                use_occurrence_weight_neg = True
                if use_occurrence_weight_neg:
                    not_in_ans_weights = 0
                    for ow in range(tagNum):
                        if ow not in feature_ans:
                            not_in_ans_weights += occurrence_weight_test_ned[ow]
                if use_occurrence_weight_pos:
                    in_ans_weights = 0
                    for ow in range(tagNum):
                        if ow in feature_ans:
                            in_ans_weights += occurrence_weight_test_ned[ow]
                for j in range(feature_length):
                    for ans_label in range(tagNum):
                        if ans_label in feature_ans:
                            if use_occurrence_weight_neg:
                                ans_list[s][i + j][ans_label] += \
                                    pos * (occurrence_weight_test_ned[ans_label] / in_ans_weights)
                            else:
                                ans_list[s][i + j][ans_label] += pos * pos_gather_in_middle_rate[j]
                        else:
                            if use_occurrence_weight_neg:
                                ans_list[s][i + j][ans_label] += \
                                    neg * (occurrence_weight_train_ned[ans_label] / not_in_ans_weights)
                            else:
                                ans_list[s][i + j][ans_label] += neg
                sub_ans_list_idx += 1
        print('sub_ans_list_idx is %d' % sub_ans_list_idx)
        print('sub_ans_list_idx has %d' % len(sub_ans_list[0]))


        out_predict = []
        total_word = 0
        nf = [0 for _ in range(tagNum)]
        pf = [0 for _ in range(tagNum)]
        nt = [0 for _ in range(tagNum)]
        pt = [0 for _ in range(tagNum)]
        for s in range(sent_num):
            out_predict_sent = [_.index(max(_)) for _ in ans_list[s]]
            for i in range(len(out_predict_sent)):
                if out_predict_sent[i] == 0:
                    out_predict_sent[i] = 1
            out_predict.append(out_predict_sent)

            def label_id_to_substring_label_id(label_id):
                substring_label_id = []
                for id in label_id:
                    if id == 0:
                        print('unk id in substring_label_id')
                    if id == 1:
                        substring_label_id.append(1)
                    else:
                        id_name = data.label_alphabet.instances[id-1]
                        label_id_name = id_name.split('-')[1]
                        substring_label_id.append(data.substring_label_alphabet.instance2index[label_id_name])
                return substring_label_id

            output = label_id_to_substring_label_id(instances[s][-1])
            for pre, tru in zip(out_predict[s], output):
                total_word += 1
                for k in range(tagNum):
                    if tru != k:
                        if pre != k:
                            nf[k] += 1
                        else:
                            pf[k] += 1
                    else:
                        if pre != k:
                            nt[k] += 1
                        else:
                            pt[k] += 1

        p = [0 for _ in range(tagNum)]
        r = [0 for _ in range(tagNum)]
        a = [0 for _ in range(tagNum)]
        f = [0 for _ in range(tagNum)]
        b = 0.00000001
        for k in range(tagNum):
            p[k] = pt[k] / (pt[k] + pf[k] + b)
            r[k] = pt[k] / (pt[k] + nt[k] + b)
            a[k] = (pt[k] + nf[k]) / (total_word + b)
            f[k] = 2 * p[k] * r[k] / (p[k] + r[k] + b)
            print("\t%dacc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (k, a[k], p[k], r[k], f[k]))
        print("\t*avg: %.4f, p: %.4f, r: %.4f, f: %.4f" % ((a[2]+a[3]+a[4]+a[5])/4,
            (p[2] + p[3] + p[4] + p[5])/4, (r[2] + r[3] + r[4] + r[5])/4, (f[2] + f[3] + f[4] + f[5])/4))

    return out_predict, ans_list


def load_model_decode(data, name):
    print "Load Model from file: ", data.model_dir
    model = SeqModel(data)
    # load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model = torch.load(model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..." % (name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return pred_results, pred_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Low Resource NER via Cross-lingual Knowledge-Transfer')
    parser = init_parser(parser)

    args = parser.parse_args()
    data = Data()
    data.load(args.data_bin_dir)
    data.read_config(args.config)
    data.read_arg(args)

    status = data.status.lower()  # train or test
    data.gpu = torch.cuda.is_available() and data.gpu

    print "Seed num:", data.seed_num
    if data.seed_num==None:
        import random
        data.seed_num = random.randint(1,100000)
        print "Random Seed num:", data.seed_num
    random.seed(data.seed_num)
    torch.manual_seed(data.seed_num)
    np.random.seed(data.seed_num)

    data.show_data_summary()

    if status == 'train':
        print("MODEL: train")
        train(data)
    elif status == 'decode_prepare':
        print("MODEL: decode_prepare")
        decode_prepare(data, searchmode=data.decode_prepare_mode)
    elif status == 'substring_viterbi_prepare':
        print("MODEL: substring_viterbi_prepare")
        substring_viterbi_prepare(data)
    elif status == 'substring_viterbi':
        print("MODEL: substring_viterbi")
        substring_viterbi(data, searchmode=data.decode_prepare_mode, prepared_data_path='')
    elif status == 'train_circulation_bilstm':
        print("MODEL: train_circulationBiLSTM")
        train_circulationBiLSTM(data)
    elif status == 'classification':
        # for pos or word
        print("MODEL: classification")
        classification(data, data.feature_name, data.feature_length, data.class_num, data.feature_ans)
    elif status == 'classification_mix':
        # for word-pos
        print("MODEL: classification_mix")
        classification_mix(data, data.feature_name, data.feature_length, data.class_num, data.feature_ans)
    elif status == 'classification_get_scores':
        # 对各个checkpoint重新测分
        print("MODEL: classification_get_scores")
        ans_dict = classification_get_scores(data, 'test', topN=1, threshold='0.5')
    elif status == 'classification_get_scores_mix':
        # 对各个checkpoint重新测分
        print("MODEL: classification_get_scores_mix")
        ans_dict = classification_get_scores_mix(data, 'test', topN=15, threshold='0.5')
    elif status == 'classification_decode':
        # 融合checkpoint预测类别，以得到单个效果最强的checkpoint.
        # 以不再做这步，因为耗时太高，且大概一半分类器融合不提分
        print("MODEL: classification_decode")
        ans_dict = load_classification_model_decode_ver2(data, 'test', topN=1, threshold='0.5')
    elif status == 'classification_load_selected_models_and_decode':
        # 读分类模型checkpoint打分结果，并生成需要的ans_list, 已过时
        print("MODEL: classification_load_selected_models_and_decode")
        ans_dict = classification_load_selected_models_and_decode(data, 'test', topN=1, rankin=['p','r','f'])
    elif status == 'classification_get_ans_list':
        # 原名是classification_load_selected_models_and_decode_v2
        # 读分类模型checkpoint打分结果，并生成需要的ans_list
        print("MODEL: classification_get_ans_list")
        ans_dict = classification_get_ans_list(data, 'test', topN=1, rankin=['p', 'r', 'f'])
    elif status == 'classification_get_ans_list_mix':
        # 读分类模型checkpoint打分结果，并生成需要的ans_list，for word-pos
        print("MODEL: classification_get_ans_list_mix")
        ans_dict = classification_get_ans_list_mix(data, 'test', topN=1, rankin=['p', 'r', 'f'])
    elif status == 'classification_ensemble':
        print("MODEL: classification_ensemble")
        out_predict, _ = make_classification_ans_ensemble(data, 'test')
    elif status == 'classification_ensemble_selected_models':
        print("MODEL: classification_ensemble_selected_models")
        out_predict, _ = make_classification_ans_ensemble_selected_models\
            (data, 'test', models='r')
    elif status == 'classification_nbest_rules_guide':
        print("MODEL: classification_nbest_rules_guide")
        ans_ensemble = classification_nbest_rules_guide(data, 'test')
    elif status == 'classification_nbest_rules_guide_selected_models':
        print("MODEL: classification_nbest_rules_guide_selected_models")
        ###not finish this status
        ans_ensemble = classification_nbest_rules_guide_selected_models(data, 'test')
    # elif status == 'decode':
    #     print("MODEL: decode")
    #     data.read_config(args.config)
    #     print data.raw_dir
    #     data.show_data_summary()
    #     data.generate_instance('raw')
    #     print("nbest: %s" % (data.nbest))
    #     decode_results, pred_scores = load_model_decode(data, 'raw')
    #     if data.nbest:
    #         data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
    #     else:
    #         data.write_decoded_results(decode_results, 'raw')
    else:
        print "Invalid argument! Please use valid arguments! (train/test/decode)"
