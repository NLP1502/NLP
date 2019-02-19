#  使用方法
#  python participle-jieba.py main --type='char'
import jieba
import fire
from tqdm import tqdm
import re

# r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+，、！%=…（）—【】“”‘’：《》；？'
# 不用去标点
# question_text = re.sub(r, ' ', question_text)

def main(**kwargs):
    type = 'word'
    if kwargs['type'] == 'char':
        type = 'char'
        fout = open("zhihu06.char", "w", encoding='utf-8')
    elif kwargs['type'] == 'word':
        type = 'word'
        fout = open("zhihu06.word", "w", encoding='utf-8')
    else:
        raise Exception("illegal type: <%s>" % type)
    file = kwargs['type']
    file = "../PyTorchText-master/task6data2_new/train_data.csv"
    with open(file) as f:
        lines = f.readlines()
    process_bar = tqdm(range(len(lines)))
    for i in process_bar:
        if i == 0:
            continue
        line_item = lines[i].strip().split(',')
        question_text = line_item[1].strip()
        # 英文分割是空格
        # question_text = line_item[1].replace(' ', '').strip()
        if len(question_text) == 0:
            continue
        if type == 'word':
            seg_list = jieba.cut(question_text, cut_all=False)
            word_list_raw = list(seg_list)
            word_list = []
            for _ in word_list_raw:
                if len(_.strip()) != 0:
                    word_list.append(_)
            fout.write(" ".join(word_list) + "\n")
        if type == 'char':
            fout.write(" ".join(question_text[::1]) + "\n")

        question_text = line_item[2].strip()
        if len(question_text) == 0:
            continue
        if type == 'word':
            seg_list = jieba.cut(question_text, cut_all=False)
            word_list_raw = list(seg_list)
            word_list = []
            for _ in word_list_raw:
                if len(_.strip()) != 0:
                    word_list.append(_)
            fout.write(" ".join(word_list) + "\n")
        if type == 'char':
            fout.write(" ".join(question_text[::1]) + "\n")


if __name__=="__main__":
    fire.Fire()
