import csv
import glob
import os.path
import shutil

fin = open("score.base_all_checked_used_final.csv","r", encoding='utf8')
modelpath = 'checkpoints/p0'
targetPath = 'result/usedModelFinal'
reader = csv.reader(fin)
files = glob.glob(modelpath + '/*')

if not os.path.exists(targetPath):
    os.makedirs(targetPath)

filesDic = {}
for file in files:
    filename = os.path.basename(file)
    seg = filename[::-1].split('_', 1)
    valScore = seg[0][::-1][0:7]
    fileModle = seg[1][::-1]
    # 使用全名
    # filesDic[fileModle + '_' + valScore] = filename
    # 使用短名
    filesDic[fileModle + '_' + valScore] = filename
    if fileModle == 'LSTMText_lstm_char_word':
        print(fileModle + '_' + valScore)

print('3333')
for row in reader:
    file = row[0] + '_' + row[1][0:7]
    print(file)
    filesDic[file]

    try:
        # 复制出去的文件使用全名
        # shutil.copyfile(os.path.join(modelpath, filesDic[file]), os.path.join(targetPath, filesDic[file]))
        # 复制出去的文件使用短名
        shutil.copyfile(os.path.join(modelpath, filesDic[file]), os.path.join(targetPath, file))
        print('copy finish')
    except:
        print('not in dic')
