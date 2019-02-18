import csv
import glob
import os.path
import shutil

fin = open("score.base_all_checked_used_final.csv","r", encoding='utf8')
modelpath = 'checkpoints/p0'
targetPath = 'result/base_all_checked_used_final'
reader = csv.reader(fin)
files = glob.glob(modelpath + '/*')

if not os.path.exists(targetPath):
    os.makedirs(targetPath)

filesDic = {}
for file in files:
    filename = os.path.basename(file)
    print(filename)
    filesDic[filename] = ''

for row in reader:
    file = row[0]
    print(file)


    try:
        filesDic[file]
        print('fined')
        # 复制出去的文件使用全名
        # shutil.copyfile(os.path.join(modelpath, filesDic[file]), os.path.join(targetPath, filesDic[file]))
        # 复制出去的文件使用短名
        shutil.copyfile(os.path.join(modelpath, file), os.path.join(targetPath, file))
        print('copy finish')
    except:
        print('not in dic')
