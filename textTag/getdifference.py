from config_06 import opt
import fire
import csv
import tqdm
import glob
import torch as t
import numpy as np

def main(**kwargs):
    opt.parse(kwargs)
    files = glob.glob(opt.test_result_path + '/*.Pre')
    l = []
    for file in files:
        print(file)
        f = t.load(file)
        l.append(t.load(file))
        # import ipdb;ipdb.set_trace()\
    loss = [[0 for _ in range(len(l))] for _ in range(len(l))]
    for i in range(len(l)):
        for j in range(len(l)):
            s = 0
            if i < j:
                for _ in range(opt.num_classes):
                    s += abs(l[i][_] - l[j][_])
            loss[i][j] = s
    print(loss)




if __name__ == '__main__':
    fire.Fire()