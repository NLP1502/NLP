# encoding:utf-8
import glob
import torch as t
import numpy as np
import os
import shutil
from config_enhance3 import opt
import models
import json
import fire
import csv
import tqdm
from torch.autograd import Variable

def load_data(type_='char'):
    with open(opt.labels_path) as f:
        labels_ = json.load(f)
    print("data_path: ", opt.test_data_path)
    question_d = np.load(opt.test_data_path)
    if type_ == 'char':
        test_data_title, test_data_content = \
            question_d['title_char'], question_d['content_char']

    elif type_ == 'word':
        test_data_title, test_data_content = \
            question_d['title_word'], question_d['content_word']

    index2qid = question_d['index2qid'].item()
    return test_data_title, test_data_content, index2qid, labels_['id2label']

def write_csv(result, index2qid, labels):
    f = open(opt.result_path, "wa")
    csv_writer = csv.writer(f, dialect="excel")
    rows = [0 for _ in range(result.shape[0])]
    for i in range(result.shape[0]):
        row = [index2qid[i]] + [labels[str(int(i_))] for i_ in result[i]]
        rows[i] = row
    csv_writer.writerows(rows)

def dotest(model, title, content):
    title, content = Variable(t.from_numpy(title).long().cuda(), volatile=True), Variable(
        t.from_numpy(content).long().cuda(), volatile=True)
    score = model(title, content)
    probs = t.nn.functional.sigmoid(score)
    return probs.data.cpu().numpy()

def main(**kwargs):
    opt.parse(kwargs)
    # import ipdb;ipdb.set_trace()
    files = glob.glob(opt.test_model_path + '/*')
    total = len(files)
    now = 0
    for file in files:
        now += 1
        print('now: ' + str(now) + '/' + str(total))
        print(file)
        if not os.path.isfile(file):
            print('is path ')
            continue
        filepath, file = os.path.split(file)
        cuts = file.split('_')
        modelkind = cuts[0]
        if modelkind == 'CNNText' and cuts[1] == 'inception':
            modelkind += '_inception'
        modeltype = cuts[-2]
        opt.model_path = os.path.join(opt.test_model_path, file)
        opt.type_ = modeltype
        try:
            # import ipdb;ipdb.set_trace()
            model = getattr(models, modelkind)(opt).cuda().eval()
            if opt.model_path is not None:
                model.load(opt.model_path)
            opt.parse(kwargs)
            opt.type_ = modeltype
            opt.result_path = os.path.join(opt.test_result_path, file) + '.pth'
            model = model.eval()

            test_data_title, test_data_content, index2qid, labels = load_data(type_=opt.type_)
            Num = len(test_data_title)
            result = np.zeros((Num, 25556))
            for i in tqdm.tqdm(range(Num)):
                if i % opt.batch_size == 0 and i > 0:
                    # import ipdb;ipdb.set_trace()
                    title = np.array(test_data_title[i - opt.batch_size:i])
                    content = np.array(test_data_content[i - opt.batch_size:i])
                    result[i - opt.batch_size:i, :] = dotest(model, title, content)
            if Num % opt.batch_size != 0:
                # original
                # title=np.array(test_data_title[opt.batch_size*(Num/opt.batch_size):])
                # content=np.array(test_data_content[opt.batch_size*(Num/opt.batch_size):])
                # result[opt.batch_size*(Num/opt.batch_size):,:]=dotest(model,title,content)
                title = np.array(test_data_title[opt.batch_size * (Num // opt.batch_size):])
                content = np.array(test_data_content[opt.batch_size * (Num // opt.batch_size):])
                result[opt.batch_size * (Num // opt.batch_size):, :] = dotest(model, title, content)
            # import ipdb;ipdb.set_trace()
            print('save')
            print(opt.result_path)
            t.save(t.from_numpy(result).float(), opt.result_path)

            if not os.path.exists(os.path.join(filepath + 'dealed')):
                print('no file')
                os.makedirs(os.path.join(filepath + 'dealed'))
            print('move from -- to --')
            print(os.path.join(filepath, file))
            print(os.path.join(filepath + 'dealed'))
            shutil.move(os.path.join(filepath, file), os.path.join(filepath + 'dealed'))
            f = open('finish.getprobability.log', 'a', encoding='utf-8')
            f.write(file + '\n')
            f.close()

        except Exception as e:
            f = open('error.getprobability.log', 'a', encoding='utf-8')
            f.write(file + '\n')
            f.close()



if __name__ == '__main__':
    fire.Fire()
