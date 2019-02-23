# -*- coding: utf-8 -*-

# generate.py use batch
# not work
import pickle
from flair.data import Sentence
from flair.models import SequenceTagger

infile = '../.flair/datasets/conll_03/eng.testb'
outfile = 'substring_viterbi_prepare_flair_conll03.pkl'
add_blank = True
batch = 10
tagger = SequenceTagger.load('ner')
loss_lists = []
feats_lists = []
gold_results = []
pred_results = []
tag_dic = tagger.tag_dictionary
inters = tagger.transitions

with open(infile, 'r') as fin:
    datas = fin.readlines()
    gold_result = []
    sent_t = ''
    lines_num_whole = len(datas)
    lines_num_now = 1

    for di in datas:
        if lines_num_now % 100 == 0:
            print('deal {}/{}'.format(lines_num_now, lines_num_whole))
        lines_num_now += 1
        if len(di.strip()) != 0:
            if len(sent_t) > 0 and add_blank:
                sent_t += ' ' + di.strip().split()[0]
            else:
                sent_t += di.strip().split()[0]
            gold_result.append(di.strip().split()[-1])
        else:
            if len(sent_t) > 0:
                sentence = Sentence(sent_t)
                sent_t = ''
                sentences, loss, feature = tagger.predict(sentence)
                gold_results.append(gold_result)
                gold_result = []
                # for i in range(feats.shape[0]):
                #     feats_lists.append(out_feats[i][0:len(instance[i][0])].cpu().detach().numpy().tolist())
                feats_lists.append(feature[0].cpu().detach().numpy().tolist())
                loss_lists.append(loss)

                pred_result = []
                for st in sentence.tokens:
                    pred_result.append(st.tags['ner'].value)
                pred_results.append(pred_result)

pickle_save = [feats_lists, inters, gold_results, pred_results, tag_dic]
fout = open(outfile, 'wb')
pickle.dump(pickle_save, fout, True)
fout.close()
