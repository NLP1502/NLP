# -*- coding: utf-8 -*-
import logging
import os
import multiprocessing
from gensim.models import Word2Vec
import sys
from gensim.models.word2vec import LineSentence
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % " ".join(sys.argv))
inp = '../zh.txt'
outp1 = '../zh.model'
outp2 = '../zh.bin'
outp3 = '../zh.w2v'
model = Word2Vec(LineSentence(inp), size=30, window=5, min_count=5,
                 workers=multiprocessing.cpu_count())

model.save(outp1)
model.wv.save(outp2)
model.wv.save_word2vec_format(outp3, binary=False)


