import json
import torch as t
import tqdm
import numpy as np

def main(**kwargs):
    inDic = ""
    inData = ''
    correctDic = ""
    outData = ""
    for k, v in kwargs.items():
        if k == 'inDic':
            inDic = v
        if k == 'inData':
            inData = v
        if k == 'correctDic':
            correctDic = v
        if k == 'outData':
            outData = v
        if k == 'test_data_path':
            test_data_path = v

    oldDic = json.load(open(inDic, 'r', encoding='utf8'))
    corDic = json.load(open(correctDic, 'r', encoding='utf8'))

    midDic = {}
    for k_,v_ in corDic['id2label'].items() :
        listSeg = v_.split(',')
        for ls in listSeg:
            midDic[ls.strip()] = k_

    oldid2label = oldDic['id2label']
    qid2label = oldDic['d']

    f = open('outlook','w', encoding='utf8')
    for key, value in oldDic['label2id'].items():
        f.write(key+ '\t' + str(value) + '\n')
    f.close()
    print('done oiddic')

    index2qid = np.load(test_data_path)['index2qid'].item()
    r = t.load(inData)
    true_labels = []
    for ii in range(len(r)):
        true_labels.append(qid2label[index2qid[ii]])


    # import ipdb;ipdb.set_trace()
    corData = []
    for rline in tqdm.tqdm(r):
        corDataLine = []
        for index in range(25551):
            # import ipdb; ipdb.set_trace()
            listSeg = corDic['id2label'][str(index)].split(',')
            pre = 0
            for seg in listSeg:
                pre += rline[oldDic['label2id'][seg]]
            pre /= len(listSeg)
            corDataLine.append(pre)
        corData.append(corDataLine)

    a = 0
    r = t.from_numpy(np.array(corData)).float()
    t.save(r, outData)

if __name__=='__main__':
    import fire
    fire.Fire()