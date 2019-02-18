import jieba
from tqdm import tqdm
import pickle
import fire
import csv

def main(**kwargs):
    infile = ""
    wordw2vfile = ""
    charw2vfile = ""
    outfile = ""
    min_count = 5

    for k, v in kwargs.items():
        if k == 'infile':
            infile = v
        if k == 'wordw2vfile':
            wordw2vfile = v
        if k == 'charw2vfile':
            charw2vfile = v
        if k == 'outfile':
            outfile = v
        if k == 'in_word2id_dic':
            in_word2id_dic = v
        if k == 'in_char2id_dic':
            in_char2id_dic = v

    word2id = pickle.load(open(in_word2id_dic, "rb"))
    char2id = pickle.load(open(in_char2id_dic, "rb"))

    with open(infile, 'r', encoding='utf8') as fin, open(wordw2vfile, 'r', encoding='utf8') as fword, open(charw2vfile, 'r', encoding='utf8') as fchar, open(outfile, 'w', encoding='utf8') as fout:
        # word_frequency = pickle.load(open("./word_frequency.dic.test", "rb"))
        # char_frequency = pickle.load(open("./char_frequency.dic.test", "rb"))

        data = fin.readlines()
        outdata = []
        process_bar = tqdm(range(len(data)))
        for i in process_bar:
            # print(data[i].replace('\t', '@@@@@@@@'))

            a, b, c, d, e = data[i].replace('\n', '').split('\t')
            b, c, d, e = [_.split(',') for _ in [b, c, d, e]]
            b2 = []
            c2 = []
            d2 = []
            e2 = []
            for w in b:
                try:
                    b2.append(char2id[w])
                except:
                    continue
            for w in c:
                try:
                    c2.append(word2id[w])
                except:
                    continue
            for w in d:
                try:
                    d2.append(char2id[w])
                except:
                    continue
            for w in e:
                try:
                    e2.append(word2id[w])
                except:
                    continue
            outdata.append([a, b2, c2, d2, e2])

        for i in process_bar:
            fout.write(outdata[i][0] + '\t' + ",".join(outdata[i][1]) + '\t' + ",".join(outdata[i][2]) + '\t' + ",".join(outdata[i][3]) + '\t' + ",".join(outdata[i][4]) + "\n")



if __name__=="__main__":
    fire.Fire()