import csv
import glob
import os.path

fout = open('select_checked.csv', 'w', encoding='utf8')
fin = open("select.csv","r", encoding='utf8')
modelpath = 'checkpoints/p0'
reader = csv.reader(fin)
files = glob.glob(modelpath + '/*')

filesDic = {}
for file in files:
    filename = os.path.basename(file)
    seg = filename[::-1].split('_', 1)
    valScore = seg[0][::-1][0:8]
    fileModle = seg[1][::-1]
    filesDic[fileModle + '_' + valScore] = ''
    print(fileModle + valScore)

for row in reader:
    file = row[0] + '_' + row[1][0:8]
    print(file)
    try:
        filesDic[file]
        fout.write(row[0] + ',' + row[1] + ',' + row[2] + '\n')
    except:
        print('not in dic')
