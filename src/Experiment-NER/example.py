# -*- coding: utf-8 -*-

#import requests
#
#url = 'http://192.168.200.169:9000'
#properties = {'annotators': 'tokenize,ssplit,pos', 'outputFormat': 'json'}
#
## properties 要转成字符串, requests包装URL的时候貌似不支持嵌套的dict
#params = {'properties' : str(properties)}
#
#data = '天气非常好'.encode("utf8")
#
#resp = requests.post(url, data, params=params)

from stanfordcorenlp import StanfordCoreNLP
# nlp=StanfordCoreNLP(r'./stanford-corenlp-full-2018-02-27/',lang='en')
nlp=StanfordCoreNLP(r'./stanford-corenlp-full-2018-02-27/',lang='en')

sentence = 'I love China'
print ('Tokenize:', nlp.word_tokenize(sentence))
print ('Part of Speech:', nlp.pos_tag(sentence))
print ('Named Entities:', nlp.ner(sentence))
print ('Constituency Parsing:', nlp.parse(sentence))#语法树
print ('Dependency Parsing:', nlp.dependency_parse(sentence))#依存句法
nlp.close() # Do not forget to close! The backend server will consume a lot memery