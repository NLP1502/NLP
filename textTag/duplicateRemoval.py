from tqdm import tqdm
import os
infile="./37700000_52000_4.txt"
trainfile="./task6data2"
with open(infile, 'r', encoding='utf8') as fin, open(trainfile, 'r', encoding='utf8') as ftrain:
    print('loading training data set')
    trains = ftrain.readlines()
    find = dict()
    for _ in tqdm(range(len(trains))):
        find[trains[_].strip().replace('/t',' ')] = ''
    data = fin.readlines()
    t = 0
    c = 0
    print('checking')
    for _ in tqdm(range(len(data))):
        t += 1
        title = data[_].split('\t',1)[0]
        if title.strip().replace('/t',' ') in find:
            c += 1
    print("repeat rate: %.2f%%" % (c/t * 100))
