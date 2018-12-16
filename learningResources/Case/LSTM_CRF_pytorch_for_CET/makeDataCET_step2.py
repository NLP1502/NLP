# -*- coding: utf-8 -*-
def changeText(file):
    fin = open(file, 'r')
    datas = fin.readlines()
    fin.close()
    fout = open(file+'.c', 'w')
    for data in datas:
        data = data[0] + ' '+ data[1] + ' ' + 'X' + ' ' + data[3] + ' '+ data[4] +'\n'
        fout.write(data)
    fout.close()

def splitText(file):
    fin = open(file, 'r')
    datas = fin.readlines()
    fin.close()
    fout = open(file + '.c', 'w')
    for data in datas:
        data = data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + data[3] + ' ' + data[4] + '\n'
        fout.write(data)
    fout.close()

changeText('trainCET.src')
changeText('validCET.src')
splitText('trainCET.tgt')
splitText('validCET.tgt')