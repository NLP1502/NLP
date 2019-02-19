# -*- coding: utf-8 -*-
import sys
import jieba
import jieba.analyse
import jieba.posseg


def turn_to(infile, outfile):
    inf = open(infile, 'r')
    outf = open(outfile, 'w')
    indata = inf.readlines()

    sen = ''
    buf = []
    buff = []
    for d in indata:
        dc = d.strip()
        if len(dc) == 0:
            if len(sen) != 0:
                sentence_seged = jieba.posseg.cut(sen.strip())
                for w in sentence_seged:
                    for c in w:
                        buff.append(w.flag)

                for i in range(len(buf)):
                    outf.write(buf[i][0] + ' ' + buff[i] + ' ' + buf[i][1] + '\n')
                outf.write('\n')
                sen = ''
                buf = []
                buff = []
        else:
            if sen == '':
                sen = dc.split()[0]
            else:
                sen += ' ' + dc.split()[0]
            buf.append(dc.split())



if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    turn_to(infile, outfile)