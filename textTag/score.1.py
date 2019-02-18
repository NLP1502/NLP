import torch as t
import numpy as np
import json
import pickle

# result = t.load("result/LSTMText0.3488~_word_test.pth")
# aes = result.numpy()
# # a = [_.split(" ") for _ in result.numpy().strip().split("\n")]
# ["title_char"]
word2id  = pickle.load(open("./word2id.dic", "rb"))
char2id = np.load('test_06.npz')

wqer = json.load(open("labels_06.json"))

aw= 1
