# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class TransBiLSTM(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout, pretrain_trans_embedding, gpu,
                 bidirect_flag=True):
        super(TransBiLSTM, self).__init__()
        print "Build translation sequence feature extractor: LSTM ..."
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        if bidirect_flag:
            self.hidden_dim = hidden_dim // 2
        self.trans_drop = nn.Dropout(dropout)
        self.trans_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_trans_embedding is not None:
            self.trans_embeddings.weight.data.copy_(torch.from_numpy(pretrain_trans_embedding))
        else:
            self.trans_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))

        self.trans_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=bidirect_flag)
        if self.gpu:
            self.trans_drop = self.trans_drop.cuda()
            self.trans_embeddings = self.trans_embeddings.cuda()
            self.trans_lstm = self.trans_lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        batch_size = input.size(0)
        trans_embeds = self.trans_drop(self.trans_embeddings(input))
        trans_hidden = None
        pack_input = pack_padded_sequence(trans_embeds, seq_lengths, True)
        trans_rnn_out, trans_hidden = self.trans_lstm(pack_input, trans_hidden)
        trans_rnn_out, trans_rnn_length = pad_packed_sequence(trans_rnn_out)
        return trans_hidden[0].transpose(1, 0).contiguous().view(batch_size, -1), trans_rnn_length

    def get_all_hiddens(self, input, seq_lengths):
        batch_size = input.size(0)
        trans_embeds = self.trans_drop(self.trans_embeddings(input))
        trans_hidden = None
        pack_input = pack_padded_sequence(trans_embeds, seq_lengths, True)
        trans_rnn_out, trans_hidden = self.trans_lstm(pack_input, trans_hidden)
        trans_rnn_out, _ = pad_packed_sequence(trans_rnn_out)
        return trans_rnn_out.transpose(1, 0)

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)
