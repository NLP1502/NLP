#coding:utf8
'''
从trian.npz中提取最后的200000条作为验证集
'''

import numpy as np
train = np.load('train_06.npz')
content_word=train['content_word'][-200000:]
title_word = train['title_word'][-200000:]
title_char = train['title_char'][-200000:]
content_char = train['content_char'][-200000:]

index2qid = train['index2qid']
np.savez_compressed('val_06.npz',
                        title_char = title_char,
                        title_word = title_word, \
                        content_char = content_char,
                        content_word = content_word,
                        index2qid = index2qid
            )