import csv
import glob
import os.path

fout = open('score.base_all_checked.csv', 'w', encoding='utf8')
fin = open("score.base_all.csv","r", encoding='utf8')
modelpath = 'checkpoints/p0'
reader = csv.reader(fin)
files = glob.glob(modelpath + '/*')

filesDic = {}
for file in files:
    filename = os.path.basename(file)
    filesDic[filename] = ''
    print(file)

for row in reader:
    file = row[0]
    print(file)
    try:
        filesDic[file]
        fout.write(row[0] + ',' + row[1] + '\n')
    except:
        print('not in dic')
