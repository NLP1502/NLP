# -*- coding: utf-8 -*-
import sys
import jieba

def turn_to(infile, outfile):
    inf = open(infile, 'r')
    outf = open(outfile, 'w')
    indata = inf.readlines()


    for d in indata:
        sen_cut = []
        cut_l = jieba.lcut(d.strip())
        for cl in cut_l:
            if len(cl.strip()) > 0:
                sen_cut.append(cl.strip())
        sen = ' '.join(sen_cut)
        outf.write(sen+'\n')


if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    turn_to(infile, outfile)