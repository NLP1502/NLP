# -*- coding: utf-8 -*-


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from wordsequence import WordSequence
from wordsequence import WordSequence_circulationBiLSTM
from crf import CRF

class SeqModel_circulationBiLSTM(nn.Module):
    def __init__(self, data, circul_time, deepth):
        super(SeqModel_circulationBiLSTM, self).__init__()
        self.use_crf = data.use_crf
        self.use_trans = data.use_trans
        self.use_mapping = data.use_mapping
        print "build network..."
        print "use_char: ", data.use_char
        if data.use_char:
            print "char feature extractor: ", data.char_seq_feature

        print "use_trans: ", data.use_trans
        print "word feature extractor: ", data.word_feature_extractor
        print "use crf: ", self.use_crf

        self.gpu = data.gpu
        self.average_batch = data.average_batch_loss
        # add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2

        self.word_hidden = WordSequence_circulationBiLSTM(data, circul_time, deepth)

        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

    def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover, batch_label, mask, trans_inputs, trans_seq_length, trans_seq_recover):
        outs, w_word_embs, trans_features_wc = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths,
                                                                char_inputs, char_seq_lengths,
                                                                char_seq_recover, trans_inputs, trans_seq_length,
                                                                trans_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        wc_loss = 0
        if self.use_trans:
            if self.use_crf:
                total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
                scores, tag_seq = self.crf._viterbi_decode(outs, mask)
                if self.use_mapping:
                    wc_loss = torch.norm(w_word_embs - trans_features_wc)
            else:
                loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
                outs = outs.view(batch_size * seq_len, -1)
                score = F.log_softmax(outs, 1)
                total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
                _, tag_seq = torch.max(score, 1)
                tag_seq = tag_seq.view(batch_size, seq_len)
                if self.use_mapping:
                    wc_loss = torch.norm(w_word_embs - trans_features_wc)
        else:
            if self.use_crf:
                total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
                scores, tag_seq = self.crf._viterbi_decode(outs, mask)
            else:
                loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
                outs = outs.view(batch_size * seq_len, -1)
                score = F.log_softmax(outs, 1)
                total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
                _, tag_seq = torch.max(score, 1)
                tag_seq = tag_seq.view(batch_size, seq_len)

        if self.average_batch:
            total_loss = total_loss / batch_size
            if self.use_mapping:
                wc_loss = wc_loss / batch_size

        return total_loss, tag_seq, wc_loss

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                mask, trans_inputs, trans_seq_length, trans_seq_recover):
        # outs:(after hidden) [batch * seq_len * label_size]
        outs, w_word_embs, trans_features_wc = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths,
                                                                char_inputs, char_seq_lengths,
                                                                char_seq_recover, trans_inputs, trans_seq_length,
                                                                trans_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)  # [batch_size * seq_len,label_size]
            _, tag_seq = torch.max(outs, 1)  # tag_seq:[batch_size * seq_len , 1] range from 0 to label_size-1
            tag_seq = tag_seq.view(batch_size, seq_len)  # [batch_size,seq_len]
            # print "before mask:{}".format(tag_seq)
            # print "mask:{}".format(mask)

            # filter padded position with zero
            tag_seq = mask.long() * tag_seq

        return tag_seq  # [batch_size,seq_len] and padding part is zero

    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover, mask, nbest, trans_inputs, trans_seq_length, trans_seq_recover):
        if not self.use_crf:
            print "Nbest output is currently supported only for CRF! Exit..."
            exit(0)
        outs, w_word_embs, trans_features_wc = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths,
                                                                char_inputs, char_seq_lengths,
                                                                char_seq_recover, trans_inputs, trans_seq_length,
                                                                trans_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.use_crf = data.use_crf
        self.use_trans = data.use_trans
        self.use_mapping = data.use_mapping
        print "build network..."
        print "use_char: ", data.use_char
        if data.use_char:
            print "char feature extractor: ", data.char_seq_feature

        print "use_trans: ", data.use_trans
        print "word feature extractor: ", data.word_feature_extractor
        print "use crf: ", self.use_crf

        self.gpu = data.gpu
        self.average_batch = data.average_batch_loss
        # add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2

        self.word_hidden = WordSequence(data)

        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

    def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover, batch_label, mask, trans_inputs, trans_seq_length, trans_seq_recover):
        outs, w_word_embs, trans_features_wc = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths,
                                                                char_inputs, char_seq_lengths,
                                                                char_seq_recover, trans_inputs, trans_seq_length,
                                                                trans_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        wc_loss = 0
        if self.use_trans:
            if self.use_crf:
                total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
                scores, tag_seq = self.crf._viterbi_decode(outs, mask)
                if self.use_mapping:
                    wc_loss = torch.norm(w_word_embs - trans_features_wc)
            else:
                loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
                outs = outs.view(batch_size * seq_len, -1)
                score = F.log_softmax(outs, 1)
                total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
                _, tag_seq = torch.max(score, 1)
                tag_seq = tag_seq.view(batch_size, seq_len)
                if self.use_mapping:
                    wc_loss = torch.norm(w_word_embs - trans_features_wc)
        else:
            if self.use_crf:
                total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
                scores, tag_seq = self.crf._viterbi_decode(outs, mask)
            else:
                loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
                outs = outs.view(batch_size * seq_len, -1)
                score = F.log_softmax(outs, 1)
                total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
                _, tag_seq = torch.max(score, 1)
                tag_seq = tag_seq.view(batch_size, seq_len)

        if self.average_batch:
            total_loss = total_loss / batch_size
            if self.use_mapping:
                wc_loss = wc_loss / batch_size

        return total_loss, tag_seq, wc_loss

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                mask, trans_inputs, trans_seq_length, trans_seq_recover):
        # outs:(after hidden) [batch * seq_len * label_size]
        outs, w_word_embs, trans_features_wc = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths,
                                                                char_inputs, char_seq_lengths,
                                                                char_seq_recover, trans_inputs, trans_seq_length,
                                                                trans_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:

            scores, tag_seq = self.crf._viterbi_decode(outs, mask)

        else:
            outs = outs.view(batch_size * seq_len, -1)  # [batch_size * seq_len,label_size]
            _, tag_seq = torch.max(outs, 1)  # tag_seq:[batch_size * seq_len , 1] range from 0 to label_size-1
            tag_seq = tag_seq.view(batch_size, seq_len)  # [batch_size,seq_len]
            # print "before mask:{}".format(tag_seq)
            # print "mask:{}".format(mask)

            # filter padded position with zero
            tag_seq = mask.long() * tag_seq

        return tag_seq  # [batch_size,seq_len] and padding part is zero

    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover, mask, nbest, trans_inputs, trans_seq_length, trans_seq_recover):
        if not self.use_crf:
            print "Nbest output is currently supported only for CRF! Exit..."
            exit(0)
        outs, w_word_embs, trans_features_wc = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths,
                                                                char_inputs, char_seq_lengths,
                                                                char_seq_recover, trans_inputs, trans_seq_length,
                                                                trans_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

    def decode_output_intermediate_result(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                mask, trans_inputs, trans_seq_length, trans_seq_recover):
        outs, w_word_embs, trans_features_wc = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths,
                                                                char_inputs, char_seq_lengths,
                                                                char_seq_recover, trans_inputs, trans_seq_length,
                                                                trans_seq_recover)
        return outs, self.crf.transitions
