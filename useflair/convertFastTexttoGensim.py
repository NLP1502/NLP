# -*- coding: utf-8 -*-

import os
emb_dir = './data/OntoNote4NER/'
f = 'w2v'
f_name = os.path.join(emb_dir,f)
from gensim.models.fasttext import FastText
model = FastText.load_fasttext_format(f_name)
word_vectors = model.wv
#model = FastText.load_binary_data(f_bin_name)
from gensim.models import KeyedVectors
word_vectors.save('../data/zh.w2v.gensim')