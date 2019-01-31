# -*- coding: utf-8 -*-
import sys

def turn_to(infile, outfile):
    inf = open(infile, 'r')
    outf = open(outfile, 'w')
    indata = inf.readlines()


    for d in indata:
        sen = ''
        dc = d.strip().split()
        for k in dc:
            sen += k
        outf.write(sen+'\n')


if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    turn_to(infile, outfile)