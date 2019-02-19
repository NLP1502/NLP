# -*- coding: utf-8 -*-

import codecs
import sys
import numpy as np
from alphabet import Alphabet
import os
from functions import *
import cPickle as pickle
from scipy import spatial

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"


class Data:
    def __init__(self):
        self.substring_names = ['word', 'pos', 'char', 'bpe']
        self.substring_maxlen = 10

        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.norm_trans_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')

        self.translation_alphabet = Alphabet('translation')
        self.translation_id_format = {}

        self.feature_name = []
        self.feature_alphabets = []
        self.feature_num = len(self.feature_alphabets)
        self.feat_config = None

        self.label_alphabet = Alphabet('label', True)
        self.tagScheme = "NoSeg"  ## BMES/BIO

        self.seg = True

        ### I/O
        self.data_bin_dir = None
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None

        self.trans_dir = None

        self.decode_dir = None
        self.model_dir = None  ## model save  file
        self.load_model_dir = None  ## model load file

        self.word_emb_dir = None
        self.char_emb_dir = None
        self.trans_embed_dir = None
        self.typeinfo_dir = None

        self.feature_emb_dirs = []

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_trans_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.trans_alphabet_size = 0

        self.feature_alphabet_sizes = []
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 50
        self.char_emb_dim = 30
        self.trans_emb_dim = 100

        ###Classification
        ## Dataset Plus
        self.substring_dir = None
        self.bpe_emb_dir = None
        self.pos_emb_dir = None
        self.pretrain_bpe_embedding = None
        self.pretrain_pos_embedding = None
        self.bpe_emb_dim = 30
        self.pos_emb_dim = 30
        self.bpe_alphabet_size = 0
        self.pos_alphabet_size = 0
        self.norm_bpe_emb = False
        self.norm_pos_emb = False
        self.bpe_texts = []
        self.bpe_Ids = []
        self.pos_texts = []
        self.pos_Ids = []
        self.label_size = 0
        self.substring_train_texts = None
        self.substring_train_Ids = None
        self.substring_dev_texts = None
        self.substring_dev_Ids = None
        self.substring_test_texts = None
        self.substring_test_Ids = None
        self.substring_label_alphabet = Alphabet('substring_label', True)

        ###Networks
        self.word_feature_extractor = "LSTM"  # "LSTM"/"CNN"/"GRU"/
        self.use_char = True
        self.char_seq_feature = "CNN"  # "LSTM"/"CNN"/"GRU"/None
        self.use_trans = False
        self.use_crf = True
        self.nbest = None
        self.use_mapping = False
        self.mapping_func = None  # tanh or sigmoid

        # Training
        self.save_model = True
        self.state_training_name = 'default'
        self.average_batch_loss = False
        self.optimizer = "SGD"  # "SGD"/"Adam"
        self.status = "train"
        self.show_loss_per_batch=100
        # Hyperparameters
        self.seed_num = None
        self.cnn_layer = 4
        self.iteration = 100
        self.batch_size = 10
        self.char_hidden_dim = 50
        self.trans_hidden_dim = 50
        self.hidden_dim = 200
        self.dropout = 0.5
        self.lstm_layer = 1
        self.bilstm = True

        self.gpu = False
        self.lr = 0.015
        self.lr_decay = 0.05
        self.clip = None
        self.momentum = 0
        self.l2 = 1e-8

    def init_substring_instance(self):
        len_names = len(self.substring_names)
        self.substring_train_texts = [[[] for _ in range(self.substring_maxlen)] for _ in range(len_names)]
        self.substring_train_Ids = [[[] for _ in range(self.substring_maxlen)] for _ in range(len_names)]
        self.substring_dev_texts = [[[] for _ in range(self.substring_maxlen)] for _ in range(len_names)]
        self.substring_dev_Ids = [[[] for _ in range(self.substring_maxlen)] for _ in range(len_names)]
        self.substring_test_texts = [[[] for _ in range(self.substring_maxlen)] for _ in range(len_names)]
        self.substring_test_Ids = [[[] for _ in range(self.substring_maxlen)] for _ in range(len_names)]

    def show_data_summary(self):
        print("++" * 50)
        print("DATA SUMMARY START:")
        print(" I/O:")
        print("     Tag          scheme: %s" % (self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Word  alphabet size: %s" % (self.word_alphabet_size))
        print("     Char  alphabet size: %s" % (self.char_alphabet_size))
        print("     Label alphabet size: %s" % (self.label_alphabet_size))
        print("     Trans alphabet size: %s" % (self.trans_alphabet_size))
        print("     Word embedding  dir: %s" % (self.word_emb_dir))
        print("     Char embedding  dir: %s" % (self.char_emb_dir))
        print("     Tran embedding  dir: %s" % (self.trans_embed_dir))
        print("     Word embedding size: %s" % (self.word_emb_dim))
        print("     Char embedding size: %s" % (self.char_emb_dim))
        print("     Tran embedding size: %s" % (self.trans_emb_dim))
        print("     Norm   word     emb: %s" % (self.norm_word_emb))
        print("     Norm   char     emb: %s" % (self.norm_char_emb))
        print("     Norm   tran     emb: %s" % (self.norm_trans_emb))
        print("   Data bin file directory: %s" % (self.data_bin_dir))
        print("     Train  file directory: %s" % (self.train_dir))
        print("     Dev    file directory: %s" % (self.dev_dir))
        print("     Test   file directory: %s" % (self.test_dir))
        print("     Raw    file directory: %s" % (self.raw_dir))
        if self.typeinfo_dir:
            print("     typeinfo    directory: %s" % (self.typeinfo_dir))
        print("     Model  file directory: %s" % (self.model_dir))
        print("     Loadmodel   directory: %s" % (self.load_model_dir))
        print("     Decode file directory: %s" % (self.decode_dir))
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("     FEATURE num: %s" % (self.feature_num))
        for idx in range(self.feature_num):
            print("         Fe: %s  alphabet  size: %s" % (
                self.feature_alphabets[idx].name, self.feature_alphabet_sizes[idx]))
            print("         Fe: %s  embedding  dir: %s" % (
                self.feature_alphabets[idx].name, self.feature_emb_dirs[idx]))
            print("         Fe: %s  embedding size: %s" % (
                self.feature_alphabets[idx].name, self.feature_emb_dims[idx]))
            print("         Fe: %s  norm       emb: %s" % (
                self.feature_alphabets[idx].name, self.norm_feature_embs[idx]))
        print(" " + "++" * 20)
        print(" Model Network:")
        print("     Model        use_crf: %s" % (self.use_crf))
        print("     Model word extractor: %s" % (self.word_feature_extractor))
        print("     Model       use_char: %s" % (self.use_char))
        if self.use_char:
            print("     Model char_seq_feature: %s" % (self.char_seq_feature))
            print("     Model char_hidden_dim: %s" % (self.char_hidden_dim))
        if self.use_trans:
            print("     Model trans_hidden_dim: %s" % (self.trans_hidden_dim))
        if self.use_mapping:
            print("     Model mapping function: %s" % (self.mapping_func))
        print(" " + "++" * 20)
        print(" Training:")
        print("     show_loss_per_batch: %s" % (self.show_loss_per_batch))
        print("     save_model: %s" % (self.save_model))
        print("     state_training_name: %s" % (self.state_training_name))
        print("     Optimizer: %s" % (self.optimizer))
        print("     Iteration: %s" % (self.iteration))
        print("     BatchSize: %s" % (self.batch_size))
        print("     Average  batch   loss: %s" % (self.average_batch_loss))

        print(" " + "++" * 20)
        print(" Hyperparameters:")

        print("     Hyper        seed_num: %s" % (self.seed_num))
        print("     Hyper              lr: %s" % (self.lr))
        print("     Hyper        lr_decay: %s" % (self.lr_decay))
        print("     Hyper            clip: %s" % (self.clip))
        print("     Hyper        momentum: %s" % (self.momentum))
        print("     Hyper              l2: %s" % (self.l2))
        print("     Hyper      hidden_dim: %s" % (self.hidden_dim))
        print("     Hyper         dropout: %s" % (self.dropout))
        print("     Hyper      lstm_layer: %s" % (self.lstm_layer))
        print("     Hyper          bilstm: %s" % (self.bilstm))
        print("     Hyper             GPU: %s" % (self.gpu))
        print("DATA SUMMARY END.")
        print("++" * 50)

        print("      substring dir : %s" % (self.substring_dir))
        print("    bpe_emb_dir dir : %s" % (self.bpe_emb_dir))
        print("    pos_emb_dir dir : %s" % (self.pos_emb_dir))
        print("++" * 50)
        sys.stdout.flush()

    def make_substring_label_alphabet(self):
        for label in self.label_alphabet.instances:
            label = label.split('-')[-1]
            self.substring_label_alphabet.add(label)
        self.substring_label_alphabet.close()


    def initial_feature_alphabets(self):
        items = open(self.train_dir, 'r').readline().strip('\n').split()
        total_column = len(items)
        if total_column > 2:
            for idx in range(1, total_column - 1):
                feature_prefix = 'feature_' + str(idx)
                self.feature_alphabets.append(Alphabet(feature_prefix))
                self.feature_name.append(feature_prefix)
                print "Find feature: ", feature_prefix
        self.feature_num = len(self.feature_alphabets)
        self.pretrain_feature_embeddings = [None] * self.feature_num
        self.feature_emb_dims = [20] * self.feature_num
        self.feature_emb_dirs = [None] * self.feature_num
        self.norm_feature_embs = [False] * self.feature_num
        self.feature_alphabet_sizes = [0] * self.feature_num
        if self.feat_config:
            for idx in range(self.feature_num):
                if self.feature_name[idx] in self.feat_config:
                    self.feature_emb_dims[idx] = self.feat_config[self.feature_name[idx]]['emb_size']
                    self.feature_emb_dirs[idx] = self.feat_config[self.feature_name[idx]]['emb_dir']
                    self.norm_feature_embs[idx] = self.feat_config[self.feature_name[idx]]['emb_norm']
        # exit(0)

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0].decode('windows-1252')
                # word = pairs[0].decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                ## build feature alphabet
                for idx in range(self.feature_num):
                    feat_idx = pairs[idx + 1].split(']', 1)[-1]
                    self.feature_alphabets[idx].add(feat_idx)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        for idx in range(self.feature_num):
            self.feature_alphabet_sizes[idx] = self.feature_alphabets[idx].size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def build_alphabet_substring(self, input_file_dir):
        ## will not read lables
        input_files = os.listdir(input_file_dir)
        print input_files
        for input_file in input_files:
            plus_feature = ''
            input_file_name = os.path.split(input_file)[1]
            if 'bpe' in input_file_name:
                plus_feature = 'bpe'
            elif 'word' in input_file_name:
                plus_feature = 'word'
            if plus_feature == '':
                continue
            in_lines = open(input_file, 'r').readlines()
            for line in in_lines:
                if len(line.strip()) > 0:
                    pairs = line.strip().split('\t')
                    words = pairs[0].decode('windows-1252')
                    # word = pairs[0].decode('utf-8')
                    if self.number_normalized:
                        words = normalize_word(words)
                    labels = pairs[-1]
                    for word in words.split():
                        self.word_alphabet.add(word)
                        for char in word:
                            self.char_alphabet.add(char)
            self.word_alphabet_size = self.word_alphabet.size()
            self.char_alphabet_size = self.char_alphabet.size()


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.translation_alphabet.close()
        for idx in range(self.feature_num):
            self.feature_alphabets[idx].close()

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrained word embedding, norm: %s, dir: %s" % (self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir,
                                                                                       self.word_alphabet,
                                                                                       self.word_emb_dim,
                                                                                       self.norm_word_emb)

            if self.typeinfo_dir:
                type_info_matrix = []
                with codecs.open(self.typeinfo_dir, 'r') as typeinfo_file:
                    type_info_lines = typeinfo_file.readlines()
                    for line in type_info_lines:
                        line = line.rstrip().split()
                        for i, _ in enumerate(line):
                            line[i] = float(line[i])
                        line = np.array(line)
                        type_info_matrix.append(line)

                print("Caculate type info distribution,and concate word and type......")
                cos_res = []
                for i, word_embed in enumerate(self.pretrain_word_embedding):
                    word_type_info = []
                    if i == 0:
                        word_type_info = np.random.random(size=len(type_info_matrix))
                        cos_res.append(word_type_info)
                    else:
                        for type_info in type_info_matrix:
                            cos_sim = 1 - spatial.distance.cosine(word_embed, type_info)
                            word_type_info.append(cos_sim)
                        cos_res.append(word_type_info)
                cos_res = np.array(cos_res)
                cos_res = sigmoid(cos_res)
                self.pretrain_word_embedding = np.concatenate([self.pretrain_word_embedding, cos_res], axis=1)
                print "type info length:{}".format(len(type_info_matrix))
                self.word_emb_dim += len(type_info_matrix)
                print "new word dim is :{}".format(self.word_emb_dim)

        if self.char_emb_dir:
            print("Load pretrained char embedding, norm: %s, dir: %s" % (self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir,
                                                                                       self.char_alphabet,
                                                                                       self.char_emb_dim,
                                                                                       self.norm_char_emb)
        if self.trans_embed_dir:
            print("Load pretrained trans embedding, norm: %s, dir: %s" % (self.norm_trans_emb, self.trans_embed_dir))
            self.pretrain_trans_embedding, self.trans_emb_dim = build_chi_pretrain_embedding(self.trans_embed_dir,
                                                                                             self.translation_alphabet,
                                                                                             self.trans_emb_dim,
                                                                                             self.norm_trans_emb)
        if self.pos_emb_dir:
            print("Load pretrained trans embedding, norm: %s, dir: %s" % (self.norm_trans_emb, self.pos_emb_dir))
            self.pretrain_pos_embedding, self.bpe_emb_dim = build_chi_pretrain_embedding(self.pretrain_pos_embedding,
                                                                                             self.feature_alphabets[0],
                                                                                             self.bpe_emb_dim,
                                                                                             self.norm_trans_emb)

        for idx in range(self.feature_num):
            if self.feature_emb_dirs[idx]:
                print("Load pretrained feature %s embedding:, norm: %s, dir: %s" % (
                    self.feature_name[idx], self.norm_feature_embs[idx], self.feature_emb_dirs[idx]))
                self.pretrain_feature_embeddings[idx], self.feature_emb_dims[idx] = build_pretrain_embedding(
                    self.feature_emb_dirs[idx], self.feature_alphabets[idx], self.feature_emb_dims[idx],
                    self.norm_feature_embs[idx])

    def generate_instance(self, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(self.train_dir, self.word_alphabet, self.char_alphabet,
                                                             self.feature_alphabets, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                             self.translation_id_format)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(self.dev_dir, self.word_alphabet, self.char_alphabet,
                                                         self.feature_alphabets, self.label_alphabet,
                                                         self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                         self.translation_id_format)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(self.test_dir, self.word_alphabet, self.char_alphabet,
                                                           self.feature_alphabets, self.label_alphabet,
                                                           self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                           self.translation_id_format)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance(self.raw_dir, self.word_alphabet, self.char_alphabet,
                                                         self.feature_alphabets, self.label_alphabet,
                                                         self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                         self.translation_id_format)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def generate_instance_substring(self):
        self.init_substring_instance()
        self.make_substring_label_alphabet()
        input_files = os.listdir(self.substring_dir)
        print input_files
        for input_file in input_files:
            input_file_name = os.path.split(input_file)[1]
            input_file_dir = os.path.join(self.substring_dir, input_file_name)
            input_file_name_split = input_file_name.split('.')
            name = input_file_name_split[1]
            feature_name = input_file_name_split[2]
            f_l = int(input_file_name_split[3][3:]) #feature_len

            if feature_name == 'word':
                alphabet = self.word_alphabet
            elif feature_name == 'char':
                alphabet = self.char_alphabet
            elif feature_name == 'pos':
                alphabet = self.feature_alphabets[0]
            elif feature_name == 'bpe':
                alphabet = self.feature_alphabets[1]

            s_f_id = self.substring_names.index(feature_name) #substring_feature_id
            if name == "train":
                self.substring_train_texts[s_f_id][f_l], self.substring_train_Ids[s_f_id][f_l]\
                    = read_instance_substring(input_file_dir, alphabet, self.substring_label_alphabet, self.number_normalized)
            elif name == "testa":
                self.substring_dev_texts[s_f_id][f_l], self.substring_dev_Ids[s_f_id][f_l] \
                    = read_instance_substring(input_file_dir, alphabet, self.substring_label_alphabet, self.number_normalized)
            elif name == "testb":
                self.substring_test_texts[s_f_id][f_l], self.substring_test_Ids[s_f_id][f_l] \
                    = read_instance_substring(input_file_dir, alphabet, self.substring_label_alphabet, self.number_normalized)
            else:
                print("Error: you can only generate train/testa/testb instance! Illegal input:%s" % (name))

    def write_decoded_results(self, predict_results, name):
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, self.decode_dir))

    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def write_nbest_decoded_results(self, predict_results, pred_scores, name):
        ## predict_results : [whole_sent_num, nbest, each_sent_length]
        ## pred_scores: [whole_sent_num, nbest]
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        assert (sent_num == len(pred_scores))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx][0])
            nbest = len(predict_results[idx])
            score_string = "# "
            for idz in range(nbest):
                score_string += format(pred_scores[idx][idz], '.4f') + " "
            fout.write(score_string.strip() + "\n")

            for idy in range(sent_length):
                label_string = content_list[idx][0][idy].encode('utf-8') + " "
                for idz in range(nbest):
                    label_string += predict_results[idx][idz][idy] + " "
                label_string = label_string.strip() + "\n"
                fout.write(label_string)
            fout.write('\n')
        fout.close()
        print("Predict %s %s-best result has been written into file. %s" % (name, nbest, self.decode_dir))

    def read_config(self, config_file):
        config = config_file_to_dict(config_file)
        ## read data:
        the_item = 'data_bin_dir'
        if the_item in config:
            self.data_bin_dir = config[the_item]
        the_item = 'train_dir'
        if the_item in config:
            self.train_dir = config[the_item]
        the_item = 'dev_dir'
        if the_item in config:
            self.dev_dir = config[the_item]
        the_item = 'test_dir'
        if the_item in config:
            self.test_dir = config[the_item]
        the_item = 'trans_dir'
        if the_item in config:
            self.trans_dir = config[the_item]

        the_item = 'substring_dir'
        if the_item in config:
            self.substring_dir = config[the_item]
        the_item = 'bpe_emb_dir'
        if the_item in config:
            self.bpe_emb_dir = config[the_item]
        the_item = 'pos_emb_dir'
        if the_item in config:
            self.pos_emb_dir = config[the_item]

        the_item = 'raw_dir'
        if the_item in config:
            self.raw_dir = config[the_item]
        the_item = 'decode_dir'
        if the_item in config:
            self.decode_dir = config[the_item]
        the_item = 'model_dir'
        if the_item in config:
            self.model_dir = config[the_item]
        the_item = 'load_model_dir'
        if the_item in config:
            self.load_model_dir = config[the_item]

        the_item = 'word_emb_dir'
        if the_item in config:
            self.word_emb_dir = config[the_item]
        the_item = 'char_emb_dir'
        if the_item in config:
            self.char_emb_dir = config[the_item]
        the_item = 'trans_embed_dir'
        if the_item in config:
            self.trans_embed_dir = config[the_item]
        the_item = 'typeinfo_dir'
        if the_item in config:
            self.typeinfo_dir = config[the_item]

        the_item = 'MAX_SENTENCE_LENGTH'
        if the_item in config:
            self.MAX_SENTENCE_LENGTH = int(config[the_item])
        the_item = 'MAX_WORD_LENGTH'
        if the_item in config:
            self.MAX_WORD_LENGTH = int(config[the_item])

        the_item = 'norm_word_emb'
        if the_item in config:
            self.norm_word_emb = str2bool(config[the_item])
        the_item = 'norm_char_emb'
        if the_item in config:
            self.norm_char_emb = str2bool(config[the_item])
        the_item = 'number_normalized'
        if the_item in config:
            self.number_normalized = str2bool(config[the_item])

        the_item = 'seg'
        if the_item in config:
            self.seg = str2bool(config[the_item])
        the_item = 'word_emb_dim'
        if the_item in config:
            self.word_emb_dim = int(config[the_item])
        the_item = 'char_emb_dim'
        if the_item in config:
            self.char_emb_dim = int(config[the_item])
        the_item = 'trans_emb_dim'
        if the_item in config:
            self.trans_emb_dim = int(config[the_item])

        ## read network:
        the_item = 'use_crf'
        if the_item in config:
            self.use_crf = str2bool(config[the_item])
        the_item = 'use_char'
        if the_item in config:
            self.use_char = str2bool(config[the_item])
        the_item = 'use_trans'
        if the_item in config:
            self.use_trans = str2bool(config[the_item])
        the_item = 'use_mapping'
        if the_item in config:
            self.use_mapping = str2bool(config[the_item])
        the_item = 'mapping_func'
        if the_item in config:
            self.mapping_func = config[the_item]
        the_item = 'word_seq_feature'
        if the_item in config:
            self.word_feature_extractor = config[the_item]
        the_item = 'char_seq_feature'
        if the_item in config:
            self.char_seq_feature = config[the_item]
        the_item = 'nbest'
        if the_item in config:
            self.nbest = int(config[the_item])

        the_item = 'feature'
        if the_item in config:
            self.feat_config = config[the_item]  ## feat_config is a dict

        ## read training setting:
        the_item = 'save_model'
        if the_item in config:
            self.save_model = str2bool(config[the_item])
        the_item = 'state_training_name'
        if the_item in config:
            self.state_training_name = config[the_item]
        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]
        the_item = 'ave_batch_loss'
        if the_item in config:
            self.average_batch_loss = str2bool(config[the_item])
        the_item = 'status'
        if the_item in config:
            self.status = config[the_item]
        the_item = 'show_loss_per_batch'
        if the_item in config:
            self.show_loss_per_batch = int(config[the_item])

        ## read Hyperparameters:
        the_item = 'seed_num'
        if the_item in config:
            if config[the_item] != 'None':
                self.seed_num = int(config[the_item])
        the_item = 'cnn_layer'
        if the_item in config:
            self.cnn_layer = int(config[the_item])
        the_item = 'iteration'
        if the_item in config:
            self.iteration = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.batch_size = int(config[the_item])

        the_item = 'char_hidden_dim'
        if the_item in config:
            self.char_hidden_dim = int(config[the_item])

        the_item = 'trans_hidden_dim'
        if the_item in config:
            self.trans_hidden_dim = int(config[the_item])

        the_item = 'hidden_dim'
        if the_item in config:
            self.hidden_dim = int(config[the_item])
        the_item = 'dropout'
        if the_item in config:
            self.dropout = float(config[the_item])
        the_item = 'lstm_layer'
        if the_item in config:
            self.lstm_layer = int(config[the_item])
        the_item = 'bilstm'
        if the_item in config:
            self.bilstm = str2bool(config[the_item])

        the_item = 'gpu'
        if the_item in config:
            self.gpu = str2bool(config[the_item])
        the_item = 'learning_rate'
        if the_item in config:
            self.lr = float(config[the_item])
        the_item = 'lr_decay'
        if the_item in config:
            self.lr_decay = float(config[the_item])
        the_item = 'clip'
        if the_item in config:
            if config[the_item] == 'None':
                self.clip = None
            else:
                self.clip = float(config[the_item])
        the_item = 'momentum'
        if the_item in config:
            self.momentum = float(config[the_item])
        the_item = 'l2'
        if the_item in config:
            self.l2 = float(config[the_item])

        ###base2
        the_item = 'feature_name'
        if the_item in config:
            self.feature_name = config[the_item]
        the_item = 'feature_length'
        if the_item in config:
            self.feature_name = int(config[the_item])
        the_item = 'class_num'
        if the_item in config:
            self.feature_name = int(config[the_item])
        the_item = 'feature_ans'
        if the_item in config:
            self.feature_name = config[the_item]


    def read_arg(self, args):
        if args.data_bin_dir != None: self.data_bin_dir = args.data_bin_dir
        if args.train_dir != None: self.train_dir = args.train_dir
        if args.dev_dir != None: self.dev_dir = args.dev_dir
        if args.test_dir != None: self.test_dir = args.test_dir
        if args.trans_dir != None: self.trans_dir = args.trans_dir
        if args.word_emb_dir != None: self.word_emb_dir = args.word_emb_dir
        if args.trans_embed_dir != None: self.trans_embed_dir = args.trans_embed_dir

        if args.substring_dir != None: self.substring_dir = args.substring_dir
        if args.bpe_emb_dir != None: self.bpe_emb_dir = args.bpe_emb_dir
        if args.pos_emb_dir != None: self.pos_emb_dir = args.pos_emb_dir

        if args.model_dir != None: self.model_dir = args.model_dir
        if args.norm_word_emb != None: self.norm_word_emb = args.norm_word_emb
        if args.norm_char_emb != None: self.norm_char_emb = args.norm_char_emb
        if args.word_emb_dim != None: self.word_emb_dim = args.word_emb_dim
        if args.char_emb_dim != None: self.char_emb_dim = args.char_emb_dim
        if args.trans_emb_dim != None: self.trans_emb_dim = args.trans_emb_dim

        if args.number_normalized != None: self.number_normalized = args.number_normalized
        if args.seg != None: self.seg = args.seg

        if args.use_crf != None: self.use_crf = args.use_crf
        if args.use_char != None: self.use_char = args.use_char
        if args.use_trans != None: self.use_trans = args.use_trans

        if args.word_seq_feature != None: self.word_seq_feature = args.word_seq_feature
        if args.char_seq_feature != None: self.char_seq_feature = args.char_seq_feature

        if args.nbest != None: self.nbest = args.nbest

        if args.status != None: self.status = args.status
        if args.state_training_name != None: self.state_training_name = args.state_training_name
        if args.save_model != None: self.save_model = args.save_model
        if args.optimizer != None: self.optimizer = args.optimizer
        if args.iteration != None: self.iteration = args.iteration
        if args.batch_size != None: self.batch_size = args.batch_size
        if args.ave_batch_loss != None: self.ave_batch_loss = args.ave_batch_loss
        if args.show_loss_per_batch != None: self.show_loss_per_batch = args.show_loss_per_batch

        if args.seed_num != None: self.seed_num = args.seed_num
        if args.cnn_layer != None: self.cnn_layer = args.cnn_layer
        if args.char_hidden_dim != None: self.char_hidden_dim = args.char_hidden_dim
        if args.trans_hidden_dim != None: self.trans_hidden_dim = args.trans_hidden_dim
        if args.hidden_dim != None: self.hidden_dim = args.hidden_dim
        if args.dropout != None: self.dropout = args.dropout
        if args.lstm_layer != None: self.lstm_layer = args.lstm_layer
        if args.bilstm != None: self.bilstm = args.bilstm
        if args.learning_rate != None: self.learning_rate = args.learning_rate
        if args.lr_decay != None: self.lr_decay = args.lr_decay
        if args.momentum != None: self.momentum = args.momentum
        if args.l2 != None: self.l2 = args.l2
        if args.gpu != None: self.gpu = args.gpu
        if args.clip != None: self.clip = args.clip

        ###base2
        if args.feature_name != None: self.feature_name = args.feature_name
        if args.feature_length != None: self.feature_length = args.feature_length
        if args.class_num != None: self.class_num = args.class_num
        if args.feature_ans != None:self.feature_ans = args.feature_ans
        if args.pretrain_pos_embedding != None:self.pretrain_pos_embedding = args.pretrain_pos_embedding
        if args.bpe_emb_dim != None:self.bpe_emb_dim = args.bpe_emb_dim



    def build_translation_alphabet(self, trans_path):
        print("Creating translation alphabet......")
        with codecs.open(trans_path, 'r', "utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip().split(":")) == 2:
                    temp = line.strip().split(":", 1)
                    words = temp[1].split()
                    for word in words:
                        self.translation_alphabet.add(word.strip())
        self.trans_alphabet_size = self.translation_alphabet.size()

    def build_translation_dict(self, trans_path):
        print("Creating Id to Id translation dictionary......")
        translation_id_format_temp = {}
        with codecs.open(trans_path, 'r', "utf-8") as f:
            lines = f.readlines()
            for line in lines:
                ids = []
                if len(line.strip().split(":", 1)) == 2:
                    temp = line.strip().split(":", 1)
                    word_id = self.word_alphabet.get_index(temp[0].strip())
                    translations = temp[1].split()
                    for translation in translations:
                        ids.append(self.translation_alphabet.get_index(translation.strip()))
                    if ids == []:
                        ids = [0]
                    translation_id_format_temp[word_id] = ids

        for word in self.word_alphabet.instances:
            if self.word_alphabet.get_index(word) in translation_id_format_temp.keys():
                self.translation_id_format[self.word_alphabet.get_index(word)] = translation_id_format_temp[
                    self.word_alphabet.get_index(word)]
            else:
                self.translation_id_format[self.word_alphabet.get_index(word)] = [0]


def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":
            continue
        if "=" in line:
            pair = line.strip().split('#', 1)[0].split('=', 1)
            item = pair[0]
            if item == "feature":
                if item not in config:
                    feat_dict = {}
                    config[item] = feat_dict
                feat_dict = config[item]
                new_pair = pair[-1].split()
                feat_name = new_pair[0]
                one_dict = {}
                one_dict["emb_dir"] = None
                one_dict["emb_size"] = 10
                one_dict["emb_norm"] = False
                if len(new_pair) > 1:
                    for idx in range(1, len(new_pair)):
                        conf_pair = new_pair[idx].split('=')
                        if conf_pair[0] == "emb_dir":
                            one_dict["emb_dir"] = conf_pair[-1]
                        elif conf_pair[0] == "emb_size":
                            one_dict["emb_size"] = int(conf_pair[-1])
                        elif conf_pair[0] == "emb_norm":
                            one_dict["emb_norm"] = str2bool(conf_pair[-1])
                feat_dict[feat_name] = one_dict
                # print "feat",feat_dict
            else:
                if item in config:
                    print("Warning: duplicated config item found: %s, updated." % (pair[0]))
                config[item] = pair[-1]
    return config


def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_parser(parser):
    parser.add_argument('--config', default="./demo.train.config", help='Configuration File')

    parser.add_argument('--data_bin_dir', default="./data/data.pickle", help='prepare data File')
    parser.add_argument('--train_dir')
    parser.add_argument('--dev_dir')
    parser.add_argument('--test_dir')
    parser.add_argument('--trans_dir')
    parser.add_argument('--word_emb_dir')
    parser.add_argument('--trans_embed_dir')
    parser.add_argument('--model_dir')

    parser.add_argument('--substring_dir')
    parser.add_argument('--bpe_emb_dir')
    parser.add_argument('--pos_emb_dir')

    parser.add_argument('--norm_word_emb', type=bool)
    parser.add_argument('--norm_char_emb', type=bool)
    parser.add_argument('--word_emb_dim', type=int)
    parser.add_argument('--char_emb_dim', type=int)
    parser.add_argument('--trans_emb_dim', type=int)

    parser.add_argument('--number_normalized', type=bool)
    parser.add_argument('--seg', type=bool)

    parser.add_argument('--use_crf', type=bool)
    parser.add_argument('--use_char', type=bool)
    parser.add_argument('--use_trans', type=bool)

    parser.add_argument('--word_seq_feature')
    parser.add_argument('--char_seq_feature')

    parser.add_argument('--nbest', type=int)

    parser.add_argument('--status')
    parser.add_argument('--state_training_name')
    parser.add_argument('--save_model', type=bool)
    parser.add_argument('--optimizer')
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--ave_batch_loss', type=bool)
    parser.add_argument('--show_loss_per_batch', type=int)

    parser.add_argument('--seed_num', type=int)
    parser.add_argument('--cnn_layer', type=int)
    parser.add_argument('--char_hidden_dim', type=int)
    parser.add_argument('--trans_hidden_dim', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--lstm_layer', type=int)
    parser.add_argument('--bilstm', type=bool)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--l2', type=float)
    parser.add_argument('--gpu', type=bool)

    parser.add_argument('--clip', type=int)

    ###base2
    # feature_name = 'word'
    # feature_length = 3
    # class_num = 2
    # feature_ans = [2, 3]
    parser.add_argument('--feature_name', type=str)
    parser.add_argument('--feature_length', type=int)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--feature_ans', type=str)

    parser.add_argument('--pretrain_pos_embedding', type=str)
    parser.add_argument('--bpe_emb_dim', type=int)


    return parser