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
from utils.data import Data
from utils.data import init_parser
import os
from math import log, exp
from utils.get_score_classification import get_score_one

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