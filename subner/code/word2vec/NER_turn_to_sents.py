# -*- coding: utf-8 -*-
# 需要手动修改pos位置
import sys

def turn_to(infile, outfile):
    inf = open(infile, 'r')
    outf = open(outfile, 'w')
    indata = inf.readlines()

    sen = ''
    for d in indata:
        dc = d.strip()
        if len(dc) == 0:
            if len(sen) != 0:
                outf.write(sen + '\n')
                sen = ''
        else:
            if sen == '':
                sen = dc.split()[-1]
            else:
                sen += ' ' + dc.split()[-1]



if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    turn_to(infile, outfile)