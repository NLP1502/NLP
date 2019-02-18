import csv
import os.path
f1=open("ans112.csv","w")
with open("score.newmodelsP41.csv","r") as f:
    reader=csv.reader(f)
    for row in reader:
        p=os.path.basename(row[0])
        p1=p.split('_')
        p2=p1[-1]
        pp=p[0:-1*len(p2)-1]
        print(pp)
        f1.write(pp)
        f1.write(",")
        if (p2[-4]=='.'):
            f1.write(p2[0:-4])
        else:
            f1.write(p2)
        f1.write(",")
        f1.write(row[1])
        f1.write("\n")

