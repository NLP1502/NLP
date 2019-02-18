import jieba
from tqdm import tqdm
import pickle
import fire
import csv

def main(inwordvect = "./word_embedding_gensim", incharvect = "./char_embedding_gensim", outwordvect = "./word_embedding_gensim.txt", outcharvect = "./char_embedding_gensim.txt"):
    word2id = pickle.load(open("./word2id.dic", "rb"))
    char2id = pickle.load(open("./char2id.dic", "rb"))
    with open(inwordvect, 'r', encoding='utf8') as finw, open(incharvect, 'r', encoding='utf8') as finc, open(outwordvect, 'w', encoding='utf8') as foutw, open(outcharvect, 'w', encoding='utf8') as foutc:
        wordfile = finw.readlines()
        charfile = finc.readlines()

        firstline = 1;
        for _ in wordfile:
            if firstline == 1:
                foutw.write(_)
                firstline = 0
                continue
            seg = _.split(' ', 1)
            try:
                flag = word2id[seg[0]]
            except:
                continue
            foutw.write(flag + ' ' + seg[1])

        firstline = 1;
        for _ in charfile:
            if firstline == 1:
                foutc.write(_)
                firstline = 0
                continue
            seg = _.split(' ', 1)
            try:
                flag = char2id[seg[0]]
            except:
                continue
            foutc.write(flag + ' ' + seg[1])


if __name__=="__main__":
    fire.Fire()