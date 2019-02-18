import jieba
from tqdm import tqdm
import fire
import csv
import pickle

def main(**kwargs):
    infile = "./task6data2_new/train_data.csv"
    inEnhancement = ''
    questionfile = "./question_train_set_06_raw.txt"
    topicfile = "./question_topic_set_06_raw.txt"

    for k, v in kwargs.items():
        if k == 'infile':
            infile = v
        if k == 'inEnhancement':
            inEnhancement = v
        if k == 'questionfile':
            questionfile = v
        if k == 'topicfile':
            topicfile = v
    if inEnhancement!= '':
        finEn = open(inEnhancement, 'r', encoding='utf8')

    with open(infile, 'r', encoding='utf8') as fin, open(questionfile, 'w', encoding='utf8') as ftrain, open(topicfile, 'w', encoding='utf8') as ftopic:
        reader = csv.reader(fin)
        data = list(reader)
        process_bar = tqdm(range(len(data)))
        word_frequency = dict()
        char_frequency = dict()
        for i in process_bar:
            if i == 0:
                continue
            # print(data[i])
            output_line = data[i][0].replace('\t','') + '\t'
            output_line2 = data[i][0].replace('\t','') + '\t'
            wlist = data[i][1].replace('\t','').strip()[::1]
            for w in wlist:
                try:
                    char_frequency[w] += 1
                except:
                    char_frequency[w] = 1
            output_line += ",".join(wlist) + '\t'
            seg_list = jieba.cut(data[i][1].replace('\t',''), cut_all=False)
            word_list_raw = list(seg_list)
            word_list = []
            for _ in word_list_raw:
                if len(_.strip()) != 0:
                    word_list.append(_)
                    try:
                        word_frequency[_] += 1
                    except:
                        word_frequency[_] = 1
            output_line += ",".join(word_list) + "\t"
            wlist = data[i][2].replace('\t','').strip()[::1]
            for w in wlist:
                try:
                    char_frequency[w] += 1
                except:
                    char_frequency[w] = 1
            output_line += ",".join(wlist) + '\t'
            seg_list = jieba.cut(data[i][2].replace('\t',''), cut_all=False)
            word_list_raw = list(seg_list)
            word_list = []
            for _ in word_list_raw:
                if len(_.strip()) != 0:
                    word_list.append(_)
                    try:
                        word_frequency[_] += 1
                    except:
                        word_frequency[_] = 1
            output_line += ",".join(word_list) + "\n"
            ftrain.write(output_line)
            output_line2 += data[i][3].replace('\t','').replace('|',',') + '\n'
            ftopic.write(output_line2)
        pickle.dump(word_frequency, open("./word_frequency.dic","wb"))
        pickle.dump(char_frequency, open("./char_frequency.dic", "wb"))


if __name__=="__main__":
    main()
    # fire.Fire()
