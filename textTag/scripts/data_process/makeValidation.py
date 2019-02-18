#coding:utf8
'''
从trian.npz中提取最后的200000条作为验证集
'''

import numpy as np
import fire

def main(**kwargs):
    for k, v in kwargs.items():
        if k == 'infile':
            infile = v
        if k == 'out_validation_set':
            out_validation_set = v
        if k == 'validation_num':
            validation_num = v

    train = np.load(infile)
    content_word=train['content_word'][-validation_num:]
    title_word = train['title_word'][-validation_num:]
    title_char = train['title_char'][-validation_num:]
    content_char = train['content_char'][-validation_num:]

    index2qid = train['index2qid']
    np.savez_compressed(out_validation_set,
                            title_char = title_char,
                            title_word = title_word, \
                            content_char = content_char,
                            content_word = content_word,
                            index2qid = index2qid
                )



if __name__=="__main__":
    fire.Fire()