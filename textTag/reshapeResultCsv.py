import jieba
from tqdm import tqdm
import fire
import csv
import pickle

def main(**kwargs):
    infile = ""
    inEnhancement = ''
    questionfile = ""
    topicfile = ""
    clear = False
    addWindowlabel = False
    for k, v in kwargs.items():
        if k == 'infile':
            infile = v
        if k == 'inEnhancement':
            inEnhancement = v
        if k == 'questionfile':
            questionfile = v
        if k == 'topicfile':
            topicfile = v
        if k == 'clear':
            clear = v
        if k == 'stopwordfile':
            stopwordfile = v
        if k == 'lablefile':
            lablefile = v
        if k == 'title_w_cut':
            title_w_cut = v
        if k == 'title_c_cut':
            title_c_cut = v
        if k == 'context_w_cut':
            context_w_cut = v
        if k == 'context_c_cut':
            context_c_cut = v
        if k == 'outwordseg':
            outwordseg = v
        if k == 'outcharseg':
            outcharseg = v
        if k == 'addWindowlabel':
            addWindowlabel = v
        if k == 'addWindowlabelAtFirst':
            addWindowlabelAtFirst = v
    outws = open(outwordseg, 'w', encoding='utf8')
    outcs = open(outcharseg, 'w', encoding='utf8')
    lables = {}
    lf = open(lablefile, 'r', encoding='utf8')
    csv_reader = csv.reader(lf)
    for line in csv_reader:
        lables[line[0].strip()] = ''
    lf.close()

    stopworddict={}
    if clear :
        fstop = open(stopwordfile, 'r', encoding='utf8')
        stopdata = fstop.readlines()
        for d in stopdata:
            stopworddict[d.strip()] = ''
    dataEnh = []
    if inEnhancement!= '':
        finEn = open(inEnhancement, 'r', encoding='utf8')
        reader = csv.reader(finEn)
        dataEnh = list(reader)

    with open(infile, 'r', encoding='utf8') as fin, open(questionfile, 'w', encoding='utf8') as ftrain, open(topicfile, 'w', encoding='utf8') as ftopic:
        reader = csv.reader(fin)
        data = list(reader)
        for _ in dataEnh:
            data.append(_)
        process_bar = tqdm(range(len(data)))
        word_frequency = dict()
        char_frequency = dict()
        total_num = len(data)
        max_title_w = 0
        max_title_c = 0
        max_context_w = 0
        max_context_c = 0
        total_title_w = 0
        total_title_c = 0
        total_context_c = 0
        total_context_w = 0
        for i in process_bar:
            if i == 0:
                continue
            # print(data[i])
            output_line = data[i][0].replace('\t','') + '\t'
            output_line2 = data[i][0].replace('\t','') + '\t'

            seg_list = jieba.cut(data[i][1].replace('\t',''), cut_all=False)
            word_list_raw = list(seg_list)
            word_list = []
            for _ in word_list_raw:
                if len(_.strip()) != 0:
                    if clear:
                        if _ in stopworddict:
                            continue
                    word_list.append(_)
                    try:
                        word_frequency[_] += 1
                    except:
                        word_frequency[_] = 1
            word_list_forchar = ''
            for _ in word_list:
                word_list_forchar += _
            wlist = word_list_forchar.replace('\t','').strip()[::1]
            for w in wlist:
                try:
                    char_frequency[w] += 1
                except:
                    char_frequency[w] = 1
            if len(word_list) > max_title_w:
                max_title_w = len(word_list)
            total_title_w += len(word_list)
            if len(wlist) > max_title_c:
                max_title_c = len(wlist)
            total_title_c += len(wlist)
            output_line += ",".join(wlist[0:title_c_cut]) + '\t'
            if len(wlist) > 0: outcs.write(" ".join(wlist[0:title_c_cut]) + '\n')
            output_line += ",".join(word_list[0:title_w_cut]) + "\t"
            if len(word_list) > 0: outws.write(" ".join(word_list[0:title_w_cut]) + "\n")

            seg_list = jieba.cut(data[i][2].replace('\t', ''), cut_all=False)
            word_list_raw = list(seg_list)
            word_list = []
            for _ in word_list_raw:
                if len(_.strip()) != 0:
                    if clear:
                        if _ in stopworddict:
                            continue
                    word_list.append(_)
                    try:
                        word_frequency[_] += 1
                    except:
                        word_frequency[_] = 1
            word_list_forchar = ''
            for _ in word_list:
                word_list_forchar += _
            wlist = word_list_forchar.replace('\t','').strip()[::1]
            for w in wlist:
                try:
                    char_frequency[w] += 1
                except:
                    char_frequency[w] = 1
            if len(word_list) > max_context_w:
                max_context_w = len(word_list)
            total_context_w += len(word_list)
            if len(wlist) > max_context_c:
                max_context_c = len(wlist)
            total_context_c += len(wlist)
            output_line += ",".join(wlist[0:context_c_cut]) + '\t'
            if len(wlist) > 0: outcs.write(" ".join(wlist[0:context_c_cut]) + '\n')
            output_line += ",".join(word_list[0:context_w_cut]) + "\n"
            if len(word_list) > 0: outws.write(" ".join(word_list[0:context_w_cut]) + '\n')
            ftrain.write(output_line)
            if addWindowlabel:
                if len(data[i][3].split('|')) <= 5:
                    addLable = 'NUMOFLABLES'+ str(len(data[i][3].split('|')))
                else:
                    addLable = 'NUMOFLABLES5'
                if addWindowlabelAtFirst:
                    output_line2 += addLable + ',' + data[i][3].replace('\t', '').replace('|', ',') + '\n'
                else:
                    output_line2 += data[i][3].replace('\t', '').replace('|', ',') + ',' + addLable + '\n'
            else:
                output_line2 += data[i][3].replace('\t','').replace('|',',') + '\n'
            ftopic.write(output_line2)

        ffre = open('zhihuData/freqW.txt', 'w', encoding='utf8')
        fredata = sorted(word_frequency.items(), key=lambda d: d[1])
        for _ in fredata:
            if _[0] not in lables:
                ffre.write(_[0] +'\n')

        print(max_title_w)
        print(max_title_c)
        print(max_context_w)
        print(max_context_c)
        print(total_title_w)
        print(total_title_c)
        print(total_context_w)
        print(total_context_c)
if __name__=="__main__":
    fire.Fire()
