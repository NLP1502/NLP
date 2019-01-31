# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from wordrep import WordRep


class WordSequence_circulationBiLSTM(nn.Module):
    def __init__(self, data, circul_time, deepth):
        super(WordSequence_circulationBiLSTM, self).__init__()
        print("Build word sequence feature extractor: %s..." % data.word_feature_extractor)
        self.circul_time = circul_time
        self.deepth = deepth
        self.hidden_dim = data.hidden_dim
        self.gather_output_mode = data.circul_gather_output_mode
        # self.gather_output_mode = 'add'

        self.gpu = data.gpu
        self.use_char = data.use_char
        self.use_trans = data.use_trans
        # self.batch_size = data.batch_size
        self.droplstm = nn.Dropout(data.dropout)
        self.bilstm_flag = data.bilstm
        self.lstm_layer = data.lstm_layer
        self.wordrep = WordRep(data)

        self.input_size = data.word_emb_dim

        if self.use_char:
            self.input_size += data.char_hidden_dim
            if data.char_seq_feature == "ALL":
                self.input_size += data.char_hidden_dim
        for idx in range(data.feature_num):
            self.input_size += data.feature_emb_dims[idx]

        if self.use_trans:
            self.input_size += data.trans_hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.hidden_dim // 2
        else:
            lstm_hidden = data.hidden_dim

        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                               bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            if self.gather_output_mode == 'concat':
                lstm_hidden /= circul_time
            elif self.gather_output_mode == 'add':
                pass
            else:
                print('no such mode for gather output: {}'.format(self.gather_output_mode))

            self.lstm_stack = []
            for i in range(deepth):
                lstm_layer = []
                for j in range(circul_time):
                    if i == 0:
                        lstm_layer.append(
                            nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer,
                                    batch_first=True, bidirectional=self.bilstm_flag))
                    else:
                        lstm_layer.append(
                            nn.LSTM(self.hidden_dim, lstm_hidden, num_layers=self.lstm_layer,
                                    batch_first=True, bidirectional=self.bilstm_flag))
                self.lstm_stack.append(lstm_layer)

        elif self.word_feature_extractor == "CNN":
            # cnn_hidden = data.hidden_dim
            self.word2cnn = nn.Linear(self.input_size, data.hidden_dim)
            self.cnn_layer = data.cnn_layer
            print "CNN layer: ", self.cnn_layer
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = (kernel - 1) / 2
            for idx in range(self.cnn_layer):
                self.cnn_list.append(
                    nn.Conv1d(data.hidden_dim, data.hidden_dim, kernel_size=kernel, padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.dropout))
                self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.hidden_dim))
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            else:
                for i in range(deepth):
                    for j in range(circul_time):
                        self.lstm_stack[i][j] = self.lstm_stack[i][j].cuda()

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                trans_inputs, trans_seq_length, trans_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        # if self.use_trans = True,w_word_embs and trans_features_wc are useful
        word_represent, w_word_embs, trans_features_wc = self.wordrep(word_inputs, feature_inputs, word_seq_lengths,
                                                                      char_inputs, char_seq_lengths,
                                                                      char_seq_recover, trans_inputs,
                                                                      trans_seq_length, trans_seq_recover)

        # word_embs (batch_size, seq_len, embed_size)
        if self.word_feature_extractor == "CNN":
            word_in = F.tanh(self.word2cnn(word_represent)).transpose(2, 1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2, 1).contiguous()
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None

            def get_lstm_in(indata, num, circul_time):
                lstm_out, lstm_out_len = pad_packed_sequence(indata)

                batch_size = lstm_out.size()[1]
                for i in range(batch_size):
                    step = lstm_out_len[i] / circul_time
                    new_first_id = step * num
                    temp = lstm_out[0:new_first_id, i]
                    lstm_out[0:lstm_out_len[i]-new_first_id, i] = lstm_out[new_first_id:lstm_out_len[i], i]
                    lstm_out[lstm_out_len[i]-new_first_id:lstm_out_len[i], i] = temp
                packed_words = pack_padded_sequence(lstm_out, lstm_out_len, False)
                return packed_words

            def get_lstm_out(indata):
                if self.gather_output_mode == 'concat':
                    return torch.cat(indata, -1)
                elif self.gather_output_mode == 'add':
                    x = indata[0]
                    for i in range(1,len(indata)):
                        x = torch.add(x, indata[i])
                    return x

            for i in range(self.deepth):
                lstm_out = []
                for j in range(self.circul_time):
                    hidden = None
                    if i == 0:
                        lstm_in = get_lstm_in(packed_words, j, self.circul_time)
                        lstm_out_s, hidden = self.lstm_stack[i][j](lstm_in, hidden)
                        lstm_out_s, _ = pad_packed_sequence(lstm_out_s)
                        lstm_out.append(lstm_out_s)
                    else:
                        lstm_in_s = get_lstm_in(lstm_in, j, self.circul_time)
                        lstm_out_s, hidden = self.lstm_stack[i][j](lstm_in_s, hidden)
                        lstm_out_s, _ = pad_packed_sequence(lstm_out_s)
                        lstm_out.append(lstm_out_s)
                lstm_in = get_lstm_out(lstm_out)
                lstm_in = pack_padded_sequence(lstm_in, word_seq_lengths.cpu().numpy(), False)
            # lstm_out (seq_len, seq_len, hidden_size)
            indata_unpack, _ = pad_packed_sequence(lstm_in)
            feature_out = self.droplstm(indata_unpack.transpose(1, 0))
        # feature_out (batch_size, seq_len, hidden_size)
        outputs = self.hidden2tag(feature_out)
        return outputs, w_word_embs, trans_features_wc

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("Build word sequence feature extractor: %s..." % data.word_feature_extractor)
        self.gpu = data.gpu
        self.use_char = data.use_char
        self.use_trans = data.use_trans
        # self.batch_size = data.batch_size
        # self.hidden_dim = data.hidden_dim
        self.droplstm = nn.Dropout(data.dropout)
        self.bilstm_flag = data.bilstm
        self.lstm_layer = data.lstm_layer
        self.wordrep = WordRep(data)

        self.input_size = data.word_emb_dim

        if self.use_char:
            self.input_size += data.char_hidden_dim
            if data.char_seq_feature == "ALL":
                self.input_size += data.char_hidden_dim
        for idx in range(data.feature_num):
            self.input_size += data.feature_emb_dims[idx]

        if self.use_trans:
            self.input_size += data.trans_hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.hidden_dim // 2
        else:
            lstm_hidden = data.hidden_dim

        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                               bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                                bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "CNN":
            # cnn_hidden = data.hidden_dim
            self.word2cnn = nn.Linear(self.input_size, data.hidden_dim)
            self.cnn_layer = data.cnn_layer
            print "CNN layer: ", self.cnn_layer
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = (kernel - 1) / 2
            for idx in range(self.cnn_layer):
                self.cnn_list.append(
                    nn.Conv1d(data.hidden_dim, data.hidden_dim, kernel_size=kernel, padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.dropout))
                self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.hidden_dim))
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            else:
                self.lstm = self.lstm.cuda()

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                trans_inputs, trans_seq_length, trans_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(batch_size, sent_len, hidden_dim)
        """
        # if self.use_trans = True,w_word_embs and trans_features_wc are useful
        word_represent, w_word_embs, trans_features_wc = self.wordrep(word_inputs, feature_inputs, word_seq_lengths,
                                                                      char_inputs, char_seq_lengths,
                                                                      char_seq_recover, trans_inputs,
                                                                      trans_seq_length, trans_seq_recover)

        # word_embs (batch_size, seq_len, embed_size)
        if self.word_feature_extractor == "CNN":
            word_in = F.tanh(self.word2cnn(word_represent)).transpose(2, 1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2, 1).contiguous()
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            # lstm_out (seq_len, seq_len, hidden_size)
            feature_out = self.droplstm(lstm_out.transpose(1, 0))
        # feature_out (batch_size, seq_len, hidden_size)
        outputs = self.hidden2tag(feature_out)
        return outputs, w_word_embs, trans_features_wc
