import argparse
import sys
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--i', type=str,
                    help='the path of input file.')
parser.add_argument('--o', type=str,
                    help='the path of output file.')
args = parser.parse_args()

infile  = args.i
outfile = args.o

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.abspath(os.path.dirname(cur_path) + os.path.sep + ".")
full2halfconfig=os.path.join(cur_dir, "half_to_full.all")


def add_contrlchr(_dic):
    for c_num in range(32):
        if chr(c_num) not in _dic:
            _dic[chr(c_num)] = ' '
    _dic[chr(127)] = ' '
    return _dic

def readConfigfile():
    global full2halfconfig
    _dic = {}
    configfile = open(full2halfconfig, "r", encoding="utf-8")
    switch = False
    for line in configfile.read().split("\n"):
        if re.match("#punctuation", line):
            switch = True
        elif switch:
            if len(line.strip().split()) == 2:
                full = line[0]
                half = line[2]
            elif len(line.strip().split()) == 1:
                full = line.strip()
                half = chr(32)
            else:
                continue

            if  full == half:
                continue
            _dic[full] = half

    return _dic

def Process(line, _dic):
    halfline = ""
    for c in line:
        nchr = _dic.get(c, c)
        halfline += nchr
    return halfline

def full2half(_dic):
    global infile
    global outfile
    # infile    = open(infile,  "r", encoding="utf-8")
    # outfile = open(outfile, "w", encoding="utf-8")
    infile    = open(infile,  "r", encoding="utf-8")
    outfile = open(outfile, "w", encoding="utf-8")
    line = infile.readline().strip()
    while True:
        if not line:
            break
        halfline = Process(line, _dic)
        outfile.write(halfline + "\n")
        line = infile.readline().strip()
    infile.close()
    outfile.close()

if __name__ == "__main__":
    full2halfdict = readConfigfile()
    full2halfdict = add_contrlchr(full2halfdict)
    full2half(full2halfdict)
    
