# encoding=utf-8
import jieba
import logging
import os
import multiprocessing
import sys
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import TfidfModel
from gensim.corpora import Dictionary


def word_tokenize(inp, outp):
    print("Word tokenize start.")
    fw = open(outp, "w", encoding="utf-8")
    jieba.load_userdict("unk")
    jieba.enable_parallel()
    with open(inp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = list(set(lines))
        length = len(lines)
        sub = 0
        for line in lines:
            sub += 1
            line = line.strip()
            if line == "":
                continue
            new_sentence = " ".join(list(jieba.cut(line, cut_all=False)))
            fw.write(new_sentence + "\n")
            if sub % 1000 == 0:
                print("%d/%d line has been processed." % (sub, length))


def word2vec(inp, outp1, outp2):
    model = Word2Vec(LineSentence(inp), size=30, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)


def tfidf(inp, outp):
    with open(inp, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = lines[:10]
    dataset = [line.split() for line in lines]
    dct = Dictionary(dataset)
    corpus = [dct.doc2bow(line) for line in dataset]

    tf_idf = TfidfModel(corpus)
    dic = tf_idf.idfs
    arr = sorted(dic.items(), key=lambda k: k[1])
    with open(outp, "w", encoding="utf-8") as f:
        for item in arr:
            f.write("%s\t\t%f\n" % (dct[item[0]], item[1]))


if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    # tokenize
    sub = 1
    if sys.argv[sub] == "-h":
        print("python pipeline.py 1 inp outp 1 inp outp1 outp2 1 inp outp")
        exit(1)

    if str(sys.argv[sub]) == "1":
        # inp, outp = "wiki.simple", "wiki.simple.token"
        inp, outp = sys.argv[sub+1], sys.argv[sub+2]
        sub = sub+2
        word_tokenize(inp, outp)
    sub += 1

    if str(sys.argv[sub]) == "1":
        # inp, outp1, outp2 = "wiki.simple.token", "wiki.model", "wiki.vector"
        inp, outp1, outp2 = sys.argv[sub+1], sys.argv[sub+2], sys.argv[sub+3]
        sub += 3
        word2vec(inp, outp1, outp2)
    sub += 1

    if str(sys.argv[sub]) == "1":
        inp, outp = "wiki.simple.token", "tf_idf"
        tfidf(inp, outp)

# python pipeline.py 0 1 xxx.txt xxx.model xxxx.vector 0