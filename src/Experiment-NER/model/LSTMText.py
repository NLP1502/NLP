import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import ipdb


def kmax_pooling(x, dim, k):
    # ipdb.set_trace()
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class LSTMText(nn.Module):

    def __init__(self, data, feature_name, feature_ans, feature_length):
        super(LSTMText, self).__init__()
        self.kmax_pooling = 2
        self.input_len = feature_length
        self.hidden_size = 256
        self.linear_hidden_size = 10
        self.num_classes = 2
        # if use multi-clssification
        # self.num_classes = data.substring_label_alphabet.size()
        self.feature_name = feature_name
        self.feature_ans = feature_ans

        self.gpu = data.gpu
        self.bilstm_flag = data.bilstm
        self.lstm_layer = data.lstm_layer
        self.average_batch = data.average_batch_loss

        if feature_name == 'word':
            alphabet_size = data.word_alphabet.size()
            embedding_dim = data.word_emb_dim
        if feature_name == 'pos':
            alphabet_size = data.feature_alphabets[0].size()
            embedding_dim = data.bpe_emb_dim

        self.input_size = embedding_dim

        self.word_embedding = nn.Embedding(alphabet_size, embedding_dim)
        if data.pretrain_word_embedding is not None and feature_name == 'word':
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        elif data.pretrain_feature_embeddings is not None and feature_name == 'pos':
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_feature_embeddings[0]))

        self.droplstm = nn.Dropout(data.dropout)
        if self.bilstm_flag:
            lstm_hidden = data.hidden_dim // 2
        else:
            lstm_hidden = data.hidden_dim


        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                                bidirectional=self.bilstm_flag)

        self.hidden2tag = nn.Linear(data.hidden_dim, self.num_classes)

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.lstm = self.lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()

        self.fc = nn.Sequential(
            nn.Linear(self.kmax_pooling*self.input_len, self.linear_hidden_size),
            nn.BatchNorm1d(self.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_hidden_size, self.num_classes)
        )
 
    def forward(self, word_inputs, word_seq_lengths, mask):
        word_represent = self.word_embedding(word_inputs)

        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _= pad_packed_sequence(lstm_out)
        lstm_out = lstm_out.permute(1, 2, 0)
        # one way
        conv_out = kmax_pooling((lstm_out), 1, self.kmax_pooling)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits
        # other way
        # outs = self.hidden2tag(lstm_out)
        # batch_size = word_inputs.size(0)
        # seq_len = word_inputs.size(1)
        # outs = outs.view(batch_size * seq_len, -1)  # [batch_size * seq_len,label_size]
        # _, tag_seq = torch.max(outs, 1)  # tag_seq:[batch_size * seq_len , 1] range from 0 to label_size-1
        # tag_seq = tag_seq.view(batch_size, seq_len)  # [batch_size,seq_len]
        # tag_seq = mask.long() * tag_seq
        #
        # return tag_seq  # [batch_size,seq_len] and padding part is zero



