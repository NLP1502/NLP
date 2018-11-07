#!/usr/local/bin/python3.7
# -*- coding:utf-8 -*-

import re

# weiboNER.conll.train
# weiboNER_2nd_conll.train
# crfsuite.weiboNER.charpos.conll.train
def rbFileCount(path):
  sent, char, entity = 0, 0, 0
  enCount = {}
  with open(path, 'rb') as f:
    for line in f.readlines():
      try:
        strs = re.split('\\s+', line.decode('utf-8').strip())
      except:
        strs = ''
      finally:
        if len(strs) >= 2:
          s, t = strs[:2]
          sent = sent + s.count('。') + s.count('？') + s.count('！')
          char = char + 1
          if 'B-' in t: 
            entity = entity + 1
            t1 = t.split('-')[-1]
            enCount[t1] = enCount[t1] + 1 if t1 in enCount else 1
  print(s, t, sent, char, entity, enCount)

# finance-ner.txt med-ner.txt utf-8 True
# mil-ner.txt gbk True
# msra_train_utf16.ner cityu_train_utf16.ner utf-16 False
def fileCount(path, encoding, type=True):
  sent, char, entity = 0, 0, 0
  enCount = {}
  with open(path, 'r', encoding=encoding) as f:
    for line in f.readlines():
      if len(line.strip()) > 0:
        strs = re.split('\\s+', line.strip())
        if len(strs) == 2:
          s, t = strs[:2]
          sent = sent + s.count('。') + s.count('？') + s.count('！')
          char = char + len(s)
          if type:
            if t != 'O':
              entity = entity + 1
              enCount[t] = enCount[t] + 1 if t in enCount else 1
          else:
            if t != 'N':
              if 'B-' in t:
                entity = entity + 1
                t1 = t.split('-')[-1]
                enCount[t1] = enCount[t1] + 1 if t1 in enCount else 1
  print(s, t, sent, char, entity, enCount)

if __name__=='__main__':
  path = '/Users/zhenglei/Desktop/AAAI论文工作/统计数据/'
  fileName = 'mil-ner.txt'
  fileCount(path + fileName, 'gbk', True)
  # rbFileCount(path + fileName)