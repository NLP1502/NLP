# environment: python 2.7, torch 0.4
# Copyright 2018-YuejiaXiang, NLP Lab., Northeastern university
# reffered to Bi-LSTM CRF model provided by pytorch tutorial file
# you can read the NOTE file in the root directory to learn more
# details.
#

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from trackMemory import Track
from timeDecoration import clock
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import test_ner

import os
import re
import codecs

# not work
# is vetebi's marsk right
gradClip = 5
dictClip = 5
DROPOUT = 0.5

# global parameters
MB_SIZE = 20
EPOCHS = 50
CHECK_PER_BATCHS = 100
SEED = 5  # seed for random
SENT_LEN_MAX = 40
INF = 10000


# model parameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
LSTM_LN = 1  # LSTM num_layers

# controlling parameters
USE_EMB_FILE = False
EMB_FILE = 'w2v'
SHOW_MEMORY_TRACK = True
USE_PAD_MARK = True
USE_PAD_MARK_IN_VITERBI = True
EMB_REQUIRES_GRAD = True
CUDA_ID = 1  # gpu id for CUDA
PYNVML_ID = 2  # gpu id for pynvml package

# init
myTrack = Track(gpu_index=PYNVML_ID, ignore_zero=True)
torch.manual_seed(SEED)
device = torch.device("cuda:" + str(CUDA_ID) if torch.cuda.is_available() else "cpu")
USE_PAD_MARK_IN_VITERBI = USE_PAD_MARK and USE_PAD_MARK_IN_VITERBI

# inherent labels
START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"
UNK_TAG = "<UNK>"

# some functions to prepare data
import numpy as np

def get_vocab(src_file_name, tgt_file_name=None, file_format='sentences'):
    char2id = {PAD_TAG: 0, UNK_TAG: 1}
    id2char = {0: PAD_TAG, 1: UNK_TAG}
    tag2id = {PAD_TAG: 0, UNK_TAG: 1, START_TAG: 2, STOP_TAG: 3}
    id2tag = {0: PAD_TAG, 1: UNK_TAG, 2: START_TAG, 3: STOP_TAG}

    if file_format == 'sentences':
        f = open(src_file_name)
        src_sentences = f.readlines()
        f.close()
        f = open(tgt_file_name)
        tgt_sentences = f.readlines()
        f.close()
    elif file_format == 'pairs':
        src_sentences, tgt_sentences = read_pairs_format(src_file_name)

    if USE_EMB_FILE:
        pretrained_emb = {}
        out_emb = []
        lower = True
        f = open(EMB_FILE, 'r')
        emb_file_datas = f.readlines()
        f.close()
        for line in emb_file_datas:
            sss = line.strip().split()
            pretrained_emb[sss[0]] = [float(x) for x in sss[1:]]
        out_emb.append([0. for _ in range(EMBEDDING_DIM)])
        out_emb.append([0. for _ in range(EMBEDDING_DIM)])

    for line in src_sentences:
        line = line.strip()
        for c in line.split(" "):
            if not USE_EMB_FILE:
                if not char2id.has_key(c):
                     char2id[c] = len(char2id)
                     id2char[len(id2char)] = c
            else:
                if c in pretrained_emb and not char2id.has_key(c):
                    char2id[c] = len(char2id)
                    id2char[len(id2char)] = c
                    out_emb.append(pretrained_emb[c])


    for line in tgt_sentences:
        line = line.strip()
        for t in line.split(" "):
            if not tag2id.has_key(t):
                tag2id[t] = len(tag2id)
                id2tag[len(id2tag)] = t
    if USE_EMB_FILE:
        return char2id, id2char, tag2id, id2tag, out_emb
    else:
        return char2id, id2char, tag2id, id2tag, None

def read_pairs_format(file_name):
    print 'begin reading pairs-format'
    src_sentences = []
    tgt_sentences = []
    src_sentence = ''
    tgt_sentence = ''

    i = 0
    f = open(file_name)
    for line in f.readlines():
        i += 1
        if i % 1000 == 0:
            sys.stdout.write("deal: %d\r" % i)
            sys.stdout.flush()

        item = line.strip().split()
        if len(item) == 0:
            src_sentences.append(src_sentence)
            tgt_sentences.append(tgt_sentence)
            src_sentence = ''
            tgt_sentence = ''

        else:
            src_sentence += item[0] + ' '
            tgt_sentence += item[1] + ' '

    if len(src_sentence) > 0:
        src_sentences.append(src_sentence)
        tgt_sentences.append(tgt_sentence)

    print 'end reading pairs-format'
    return src_sentences, tgt_sentences

def get_dataset(char2id, tag2id, src_file_name, tgt_file_name=None, file_format='sentences'):
    X = []
    Y = []
    mbx = []
    mby = []
    seq_lengthsX = []
    seq_lengthsY = []

    if file_format == 'sentences':
        f = open(src_file_name)
        src_sentences = f.readlines()
        f.close()
        f = open(tgt_file_name)
        tgt_sentences = f.readlines()
        f.close()
    elif file_format == 'pairs':
        src_sentences, tgt_sentences = read_pairs_format(src_file_name)

    for line in src_sentences:
        line = line.strip()
        x = [char2id.get(c, char2id[UNK_TAG]) for c in line.split(" ")]
        mbx.append(x)
        if len(mbx) == MB_SIZE:
            mbx = sorted(mbx, key=lambda item: len(item), reverse=True)
            seq_lengthsX.append(torch.LongTensor(
                [min(x, SENT_LEN_MAX) for x in map(len, mbx)]))
            batch_length = max(seq_lengthsX[-1]).tolist()
            for i in range(len(mbx)):
                mbx[i] = mbx[i][:batch_length] + [tag2id[PAD_TAG]] \
                         * (batch_length - len(mbx[i]))
            X.append(zip(*mbx))
            mbx = []

    for line in tgt_sentences:
        line = line.strip()
        y = [tag2id.get(t, char2id[UNK_TAG]) for t in line.split(" ")]
        mby.append(y)
        if len(mby) == MB_SIZE:
            mby = sorted(mby, key=lambda item: len(item), reverse=True)
            seq_lengthsY.append(torch.LongTensor(
                [min(x, SENT_LEN_MAX) for x in map(len, mby)]))
            batch_length = max(seq_lengthsY[-1]).tolist()
            for i in range(len(mby)):
                mby[i] = mby[i][:batch_length] + [tag2id[PAD_TAG]] \
                         * (batch_length - len(mby[i]))
            Y.append(zip(*mby))
            mby = []

    return X, Y, seq_lengthsX

# some unit functions

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.cpu().numpy()

def argmax_t(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = torch.gather(vec, dim=1, index=argmax_t(vec).view(-1,1))
    max_score_broadcast = max_score.view(-1, 1).expand(MB_SIZE, vec.size()[-1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

#Create the model
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("model.init: begin")
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("model.init: Embedding")
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=LSTM_LN, bidirectional=True, dropout=DROPOUT)
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("model.init: LSTM")
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("model.init: Linear")
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size).to(device))
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("model.init: transitions")
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -INF
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -INF
        self.hidden = self.init_hidden()
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("model.init: end")

    def init_hidden(self):
        return (torch.zeros(LSTM_LN * 2, MB_SIZE, self.hidden_dim // 2).to(device),
                torch.zeros(LSTM_LN * 2, MB_SIZE, self.hidden_dim // 2).to(device))

    def _forward_alg(self, feats, seq_length):
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_forward_alg: begin")
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((MB_SIZE, self.tagset_size), -INF).to(device)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # forward_var_L = []
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # forward_var_L.append(forward_var.clone())

        # Iterate through the sentence
        for num, feat in enumerate(feats):
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_tag].view(
                    MB_SIZE, 1).expand(MB_SIZE, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(-1, 1))
            if USE_PAD_MARK:
                forward_var_new = torch.cat(alphas_t, 1).view(MB_SIZE, -1)
                for mb in range(MB_SIZE):
                    if seq_length[mb] > num:
                        forward_var[mb] = forward_var_new[mb]
            else:
                forward_var = torch.cat(alphas_t, 1).view(MB_SIZE, -1)
            if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_forward_alg: loop%d" % num)
            # forward_var_L.append(forward_var.clone())
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # forward_var_L.append(terminal_var.clone())
        alpha = log_sum_exp(terminal_var)
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_forward_alg: end")
        return alpha

    def _get_lstm_features(self, sentence, seq_length):
        # get the length of each seq in your batch
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_get_lstm_features: begin")
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), MB_SIZE, -1)
        packed_input = pack_padded_sequence(embeds, seq_length)
        packed_output, self.hidden = self.lstm(packed_input, self.hidden)
        lstm_out, _ = pad_packed_sequence(packed_output)
        lstm_out = lstm_out.view(len(sentence), MB_SIZE, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_get_lstm_features: end")
        return lstm_feats

    def _score_sentence(self, feats, tags, seq_length):
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_score_sentence: begin")
        if USE_PAD_MARK:
            padid = self.tag_to_ix['<PAD>']
        # scoreL = []
        # Gives the score of a provided tag sequence
        score = torch.zeros(MB_SIZE).to(device)
        starTag_expand = torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).\
            to(device).expand(MB_SIZE).view(1,MB_SIZE)
        tags = torch.cat([starTag_expand, tags])
        for i, feat in enumerate(feats):
            a = torch.index_select(self.transitions, 1, tags[i])
            b = torch.gather(torch.t(a), dim=1, index=tags[i+1].view(MB_SIZE, 1)).view(-1)
            c = torch.gather(feat, dim=1, index=tags[i+1].view(MB_SIZE, 1)).view(-1)
            if USE_PAD_MARK:
                for mb in range(MB_SIZE):
                    if tags[i+1][mb] == padid:
                        b[mb] = 0
                        c[mb] = 0
            score = score + b + c
            # scoreL.append(score)
            if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_score_sentence: loop%d" % i)

        if USE_PAD_MARK:
            end_tags = torch.LongTensor([0 for _ in range(MB_SIZE)]).to(device)
            for i in range(MB_SIZE):
                end_tags[i] = tags[seq_length[i]][i]
            a = torch.index_select(self.transitions, 1, end_tags)
        else:
            a = torch.index_select(self.transitions, 1, tags[-1])
        b = torch.gather(torch.t(a), dim=1, index=torch.LongTensor(
            [self.tag_to_ix[STOP_TAG]]*MB_SIZE).to(device).view(MB_SIZE, 1)).view(-1)
        score = score + b
        # scoreL.append(score)
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_score_sentence: end")
        return score

    def _viterbi_decode(self, feats, seq_length):
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_viterbi_decode: begin")
        backpointers = []
        forward_var_L = []
        if USE_PAD_MARK: bptrs_t_c = [[] for _ in range(MB_SIZE)]

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((MB_SIZE, self.tagset_size), -10000.).to(device)
        init_vvars[:, self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        forward_var_L.append(forward_var.clone())
        for num, feat in enumerate(feats):
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                # viterbivars_t.append(next_tag_var[0][best_tag_id].view(MB_SIZE, 1))
                # This line can be simplify
                viterbivars_t.append(torch.gather(next_tag_var, 1, torch.LongTensor(
                    best_tag_id).to(device).view(MB_SIZE, 1)))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            if USE_PAD_MARK_IN_VITERBI:
                forward_var_new = (torch.cat(viterbivars_t, 1) + feat).view(MB_SIZE, -1)
                bptrs_t_c = torch.t(torch.LongTensor(bptrs_t))
                for mb in range(MB_SIZE):
                    if seq_length[mb] > num:
                        forward_var[mb] = forward_var_new[mb]
                    if seq_length[mb] <= num:
                        bptrs_t_c[mb] = backpointers[-1][mb]
                backpointers.append(bptrs_t_c)
            else:
                forward_var = (torch.cat(viterbivars_t, 1) + feat).view(MB_SIZE, -1)
                bptrs_t_c = torch.t(torch.LongTensor(bptrs_t))
                backpointers.append(bptrs_t_c)
            # bptrs_t_c = torch.LongTensor(bptrs_t).view(MB_SIZE,-1)
            forward_var_L.append(forward_var.clone())
            if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_viterbi_decode: loop%d" % num)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        forward_var_L.append(terminal_var.clone())

        best_tag_id = argmax_t(terminal_var).view(-1,1)
        # print best_tag_id
        path_score = torch.gather(terminal_var, dim=1, index=best_tag_id).view(-1)

        # Follow the back pointers to decode the best path.
        # best_path = [best_tag_id]
        best_path = best_tag_id.cpu()
        for bptrs_t in reversed(backpointers):
            # best_tag_id = bptrs_t[best_tag_id]
            best_tag_id = torch.gather(bptrs_t, dim=1, index=best_tag_id.cpu())
            # best_path.append(best_tag_id)
            best_path = torch.cat([best_path, best_tag_id], 1)
        # Pop off the start tag (we dont want to return that to the caller)
        # start = best_path.pop()
        best_path_reverse = best_path.numpy()[0:MB_SIZE,0:-1]
        # assert (start.numpy() == np.array([[self.tag_to_ix[START_TAG]]*MB_SIZE])).all()  # Sanity check
        # best_path.reverse()
        # start.reverse()
        # return path_score, best_path
        for bt in range(MB_SIZE):
            best_path_reverse[bt] = best_path_reverse[bt][::-1]
        best_path = best_path_reverse
        if USE_PAD_MARK_IN_VITERBI:
            for mb in range(MB_SIZE):
                best_path[mb][seq_length[mb]-1] = best_path[mb][-1]
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("_viterbi_decode: end")
        return path_score, best_path_reverse

    def neg_log_likelihood(self, sentence, tags, seq_length):
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("neg_log_likelihood: begin")
        feats = self._get_lstm_features(sentence.to(device), seq_length)
        forward_score = self._forward_alg(feats, seq_length)
        gold_score = self._score_sentence(feats, tags.to(device), seq_length)
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("neg_log_likelihood: end")
        return forward_score.view(-1) - gold_score

    def forward(self, sentence, seq_length):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("forward: begin")
        lstm_feats = self._get_lstm_features(sentence.to(device), seq_length)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats, seq_length)
        if SHOW_MEMORY_TRACK: myTrack.show_memory_add("forward: end")
        return score, tag_seq

# Train the model:

acc = []
print '[' + datetime.now().strftime('%H:%M:%S') + ']', "loading data ..."
# char2id, id2char, tag2id, id2tag, out_emb = get_vocab("trainToy.src", "trainToy.tgt", file_format='sentences')
# X, Y, seq_lengths = get_dataset(char2id, tag2id, "trainToy.src", "trainToy.tgt", file_format='sentences')
# VX, VY, seq_lengths_V = get_dataset(char2id, tag2id, "validToy.src", "validToy.tgt", file_format='sentences')
char2id, id2char, tag2id, id2tag, out_emb = get_vocab("trainFin.txt", file_format='pairs')
X, Y, seq_lengths = get_dataset(char2id, tag2id, "trainFin.txt", file_format='pairs')
VX, VY, seq_lengths_V = get_dataset(char2id, tag2id, "testFin.txt", file_format='pairs')
# VX, VY, seq_lengths_V = get_dataset(char2id, tag2id, "trypairdata.txt", file_format='pairs')

# model = torch.load('model_87.50.pkl')
model = BiLSTM_CRF(len(char2id), tag2id, EMBEDDING_DIM, HIDDEN_DIM)
print(model)

if USE_EMB_FILE:
    # weights = load_word2vec(pretrained_word2vec, id2char, word_dim, self.word_embeds.)
    model.word_embeds.weight.data.copy_(torch.from_numpy(np.array(out_emb)))
if not EMB_REQUIRES_GRAD:
    model.word_embeds.weight.requires_grad = False
# for param in model.parameters():
#     print param
    # param.requires_grad = False
# model2= nn.Sequential(*list(model.children())[:-1])
# print(model2)
def get_features_hook(self, input, output):
    print(input)
    print("hook",output[0].data.cpu().numpy().shape)
addhook = False
if addhook:
    handle=(list(model.children())[1]).register_forward_hook(get_features_hook)
# handle.remove()

if SHOW_MEMORY_TRACK: myTrack.show_memory_add("main: before model.cuda()")
model.to(device)
if SHOW_MEMORY_TRACK: myTrack.show_memory_add("main: after model.cuda()")

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=0.01, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
if SHOW_MEMORY_TRACK: myTrack.show_memory_add("main: after define optimizer")

# Check predictions before training
# with torch.no_grad():
#     precheck_sent = torch.tensor(X[0], dtype=torch.long)
#     precheck_tags = torch.tensor(Y[0], dtype=torch.long)
#     print(model(precheck_sent))


max_a = 0
@clock
def main():
    for epoch in range(EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy dat
        print 'epoch %d' % (epoch+1)

        @clock
        # train step
        def train_step():
            batch_num = 1
            epoch_loss = 0
            for sentence, tags, seq_length in zip(X, Y, seq_lengths):
                if batch_num % CHECK_PER_BATCHS == 0:
                    print 'processing %d/%d' % (batch_num, len(X))
                    eval_step()
                batch_num += 1

                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                sentence_in = torch.tensor(sentence, dtype=torch.long)
                targets = torch.tensor(tags, dtype=torch.long)

                # Step 3. Run our forward pass.
                loss = model.neg_log_likelihood(sentence_in, targets, seq_length)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss_b = (loss.sum()/MB_SIZE)
                loss_b.backward()
                epoch_loss += loss_b.tolist()
                optimizer.step()
            print 'epoch loss: %f' % epoch_loss

        # evaluate step
        def eval_step():
            global max_a
            ner_results = []
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y, seq_length in zip(VX, VY, seq_lengths_V):
                    x = torch.tensor(x)
                    y = torch.tensor(y)

                    # model.hidden = model.init_hidden()
                    _, predicted = model(x, seq_length)

                    for bt in range(MB_SIZE):
                        bt_list = predicted[bt].tolist()
                        lenth = len(bt_list) - torch.t(y)[bt].numpy().tolist().count(0)
                        total += lenth
                        correct += (torch.tensor(bt_list[0:lenth]) == torch.t(y)[bt][0:lenth]).sum().item()

                        block = []
                        for c in range(lenth):
                            block.append(id2char[torch.t(x)[bt][c].tolist()] + ' ' + id2tag[bt_list[c]] + ' ' + id2tag[torch.t(y)[bt][c].tolist()])
                        ner_results.append(block)
            eval_lines = test_ner(ner_results, '')
            a = float(eval_lines[1].strip().split()[-1])

            # a = 100.0 * correct / total
            print 'acc = %.2f' % a
            if a > max_a:
                # torch.save(model, 'model' + '_%.2f' % a + '.pkl')
                max_a = a

        train_step()
        eval_step()
    print "best score: %.2f" % max_a

main()