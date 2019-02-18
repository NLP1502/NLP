import sys
sys.path.append('../')
import fire
import torch as t

def main(**kwargs):
    resultAPath = sys.argv[2]
    resultBPath = sys.argv[3]
    outfilePath = sys.argv[4]
    print(sys.argv)
    weight = int(sys.argv[5])
    print(weight)
    r = weight * t.load(resultAPath)
    r += t.load(resultBPath)
    t.save(r, outfilePath)

if __name__ == '__main__':
    fire.Fire()