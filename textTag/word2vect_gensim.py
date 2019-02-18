# encoding=utf-8
import logging
import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import fire

def main(**kwargs):
    for k, v in kwargs.items():
        print(k)
        print(v)
        if k == 'inSegFile':
            inSegFile = v
        if k == 'outVectorFile':
            outVectorFile = v
        if k == 'sizeNum':
            sizeNum = v
        if k == 'windowNum':
            windowNum = v
        if k == 'min_countNum':
            min_countNum = v
    model = Word2Vec(LineSentence(inSegFile), size=sizeNum, window=windowNum, min_count=min_countNum,
                     workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format(outVectorFile, binary=False)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    fire.Fire()
