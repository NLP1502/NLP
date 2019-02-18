import torch as t
import sys
sys.path.append('../')
import csv
import json
import os
import fire
import glob
import torch as t
import numpy as np




def main(**kwargs):
    maxTop = 1
    weight = 1.015
    ansFileName = 'greedy_final_dev/ensemble_1_1.pth'
    rankfilelist = ['greedy_dev/pth/FastText3_fasttext3_bsize0.25_enhance3_word_0.302397522172602&16&999.pth','greedy_dev/pth/LSTMText_lstm_char_e4_enhance3_char_0.28358828629949423&12&2999.pth','greedy_dev/pth/LSTMText_lstm_lhid2_enhance3_word_0.2960218893947179&11&2999.pth','greedy_dev/pth/LSTMText_lstm_word_enhance3_word_0.2980656602982555&16&2999.pth']
    # rankfilelist = ['greedy_final_test/FastText3_fasttext3_bsize0.25_enhance3_word_0.302397522172602&16&999.pth','greedy_final_test/LSTMText_lstm_char_e4_enhance3_char_0.28358828629949423&12&2999.pth','greedy_final_test/LSTMText_lstm_lhid2_enhance3_word_0.2960218893947179&11&2999.pth','greedy_final_test/LSTMText_lstm_word_enhance3_word_0.2980656602982555&16&2999.pth']


    for i in range(len(rankfilelist)):
        if i == 0:
            r = t.load(rankfilelist[i])
        else:
            r += weight * t.load(rankfilelist[i])

    t.save(r, ansFileName)



if __name__ == '__main__':
    fire.Fire()

