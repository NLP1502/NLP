# environment: python 2.7, torch 0.4
# Copyright 2017-YuejiaXiang, NLP Lab., Northeastern university
# reffered to Bi-LSTM CRF model provided by pytorch tutorial file
# you can read the README file in the root directory to learn more
# details.
#

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 20
HIDDEN_DIM = 40

SENT_LEN = 50
LSTM_LN = 2
INF = 10000

START_TAG = "<START>"
STOP_TAG = "<STOP>"
def get_vocab(src_file_name, tgt_file_name):
    char2id = {'<PAD>': 0, '<UNK>': 1}
    id2char = {0: '<PAD>', 1: '<UNK>'}
    tag2id = {'<PAD>': 0, '<UNK>': 1, START_TAG: 2, STOP_TAG: 3}
    id2tag = {0: '<PAD>', 1: '<UNK>', 2: START_TAG, 3: STOP_TAG}

    f = open(src_file_name)
    while 1:
        line = f.readline()
        if line == "": break
        line = line.strip().decode("utf-8")

        for c in line.split(" "):
            if not char2id.has_key(c):
                 char2id[c] = len(char2id)
                 id2char[len(id2char)] = c

    f.close()

    f = open(tgt_file_name)

    while 1:
        line = f.readline()
        if line == "": break
        line = line.strip().decode("utf-8")

        for t in line.split(" "):
            if not tag2id.has_key(t):
                tag2id[t] = len(tag2id)
                id2tag[len(id2tag)] = t

    f.close()

    return char2id, id2char, tag2id, id2tag

def get_dataset(src_file_name, tgt_file_name, char2id, tag2id):
    X = []
    Y = []

    f = open(src_file_name)
    while 1:
        line = f.readline()
        if line == "": break
        line = line.strip().decode("utf-8")

        x = [char2id.get(c, 1) for c in line.split(" ")]
        X.append(x)

    f.close()

    f = open(tgt_file_name)
    while 1:
        line = f.readline()
        if line == "": break
        line = line.strip().decode("utf-8")

        y = [tag2id.get(t, 1) for t in line.split(" ")]
        Y.append(y)

    f.close()

    return X, Y


######################################################################
# Create the model:
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size).to(device))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
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
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence.to(device))
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags.to(device))
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence.to(device))

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
######################################################################
# Train the model:

acc = []

print '[' + datetime.now().strftime('%H:%M:%S') + ']', "loading data ..."
char2id, id2char, tag2id, id2tag = get_vocab("train.src", "train.tgt")
X, Y = get_dataset("train.src", "train.tgt", char2id, tag2id)
VX, VY = get_dataset("valid.src", "valid.tgt", char2id, tag2id)

model = BiLSTM_CRF(len(char2id), tag2id, EMBEDDING_DIM, HIDDEN_DIM)
# model.cuda()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = torch.tensor(X[0], dtype=torch.long)
    precheck_tags = torch.tensor(Y[0], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
max_a = 0
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy dat
    print 'epoch %d' % epoch
    for sentence, tags in zip(X, Y):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = torch.tensor(sentence, dtype=torch.long)
        targets = torch.tensor(tags, dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

        i = 0
        correct = 0
        total = 0

    with torch.no_grad():
        # import ipdb
        # ipdb.set_trace()
        for x, y in zip(VX, VY):
            x = torch.tensor(x)
            y = torch.tensor(y)

            # model.hidden = model.init_hidden()

            _, predicted = model(x)

            total += len(x)
            correct += (torch.tensor(predicted) == y).sum().item()

            i += 1
            if i % 100 == 0:
                sys.stdout.write("progress:%d/%d\r" % (i, len(VX)))
                sys.stdout.flush()

    a = 100.0 * correct / total
    print 'acc = %d' % a
    if a > max_a:
        max_a = a
print max_a
