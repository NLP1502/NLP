# -*- coding: utf-8 -*-
fin = open('cet6.txt', 'r')
fout = open('dataCET', 'w')
datas = fin.readlines()
for data in datas:
    list = data.strip().split('\t')[1]
    for i in range(len(list)-4):
        fout.write(list[i:i+5] + '\n')


fin.close()
fout.close()