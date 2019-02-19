# 实验计划

## 时间安排  
计划：  
19.1.24-19.1.27：修改模型，用翻译做纠错  
19.1.28-19.1.31：NER、纠错初步初步实验  
19.2.1-19.2.10：调参实验，论文阅读  
19.2.11-19.2.15：模型修改  
19.2.16-19.2.20：论文初稿，在其他基线或任务上尝试  
19.2.21-19.2.25：论文二稿，调参实验  
19.2.26-19.2.28：论文三稿，调参实验
> 已完成：  
18.12.15-19.1.2: 整理18年顶会NER论文  
18.1.3-19.1.6：用NER模型和CRF++工具做纠错  
19.1.7-19.1.11：生成随机伪数据验证NER模型在纠错任务上的潜力  
19.1.11-19.1.17：选择基线模型并初步复现  
19.1.18：列实验计划  
19.1.19-19.1.20：调整基线模型  
19.1.20-19.1.23：基线模型调参  
  
## 实验方案
> 由于时间很紧，优先选择对模型改动较小的方案

### 基线模型调参实验
> 动机：尝试复现论文结果, 并初步确认不同参数的影响力。  
>>方法：保持基本结构不变，调整超参。 

### 增加特征
> 动机：扩展基线模型所使用的特征，初步确认不同特征对基线模型的影响力  
>>方法：多语言、POS、分词、字形、BPE

### 字概率信息
> 动机：把统计方法和深度学习方法结合起来  
>> 方法：在字级别上引入不同特征的统计概率。  
获取更适合本任务的词向量   
中文中是字，英文中是BPE字词  

<img src="https://github.com/ApricotXingyaYJ/markdownImages/blob/master/ner-experiments/1.png?raw=true" width="40%" height="40%">  

### 局部聚焦
> 动机：人往往只需看局部字符串就可以判断实体
>> 利用子符串而非全句进行NER  
在子字符串上使用循环LSTM或CNN、使用交叉验证分类器  

<img src="https://github.com/ApricotXingyaYJ/markdownImages/blob/master/ner-experiments/2.png?raw=true" width="40%" height="40%">  

### 多层LSTM
> 动机：NER任务中实体标签较稀疏且和POS强相关，提出一种更适合该任务的特征提取结构  
>> 方法：动态链接的多层lstm实现任两词的直连、或用POS做attention  

<img src="https://github.com/ApricotXingyaYJ/markdownImages/blob/master/ner-experiments/4.png?raw=true" width="100%" height="40%">  

### 决策层模型融合
> 动机：不仅仅考虑最优预测结果，较次的预测结果也许可以对提高性能有帮助
>> 方法：用模型融合的方法、结合其他附加信息

<img src="https://github.com/ApricotXingyaYJ/markdownImages/blob/master/ner-experiments/3.png?raw=true" width="40%" height="40%">  

### 特征的引入时机与方法
> 动机：不同的外部信息本身的抽象级别不同，应在不同的位置加入到模型中
>> 方法：分别在输入层、中间层、决策层和结果层尝试维度拼接、对位相加、门控系数

<img src="https://github.com/ApricotXingyaYJ/markdownImages/blob/master/ner-experiments/5.png?raw=true" width="40%" height="40%">  

# 实施记录
注意：全部实验log使用统一的命名规则，如logNer.base.a1-2.a2-1.c3-2.ra

## a# 基线模型调参实验
>现有模型：  
实验的基础模型记为base。  
修改了输入参数方式的基础模型记为base1。

+ a1# iteration=300、600
+ a2# lstm_layer=2、4、8、3
+ a3# dropout=0、0.2
+ a4# batchsize=32
+ a5# char_hidden_dim=100
+ a6# trans_hidden_dim=400
+ a7# hidden_dim=400
+ a8# learning_rate=0.008、0.03
+ a9# lr_decay=0.1
+ a10# l2=0
+ a11# clip=5(需要修改代码)
+ a12# 修改特征：notrain, noPOS

## b# 增加特征

+ b1# 更大英语trains文件: 
+ b2# 其他语trains文件
+ b3# POS
+ b4# 分词位置
+ b5# 字形特征
+ b6# 多gram的idf概率
+ b7# 使用feature_emb_dirs
+ b8# 特征引入的方法：拼接、門控

## c# 字概率信息

+ c1# 输入层硬拼接
+ c2# 输入层软拼接
+ c3# 中间层软拼接
+ c4# 决策层硬拼接
+ c5# 引入实体类型概率：统计方法、用词向量相似度学
+ c6# 引入POS概率
+ c7# 引入分词概率

## d# 局部聚焦

+ d1#方式=局部emb、LM判断
+ d2# 局部emb融入方式
+ d3# 各类型单独训练LM判断子串是否含实体
+ d4# 多类型交叉训练LM判断=2组合、全组合
+ d5#判断方式=ppl、分类器、emb拼接
+ d6# embedding拼接方式=加和、LSTM
+ d7# 评价标准=ppl、相似度、分类器
+ d8# 标签向量生成方式=加权平均、与多个分类器一起学
+ d9# 子串长度=1、2、3、4、5
+ d10# 多判断结果的综合方法
+ d11# 使用该特征的方法：输入层硬拼接、决策层硬拼接

## e# 多层lstm

+ e1# 纵向的门控单元
+ e2# attention
+ e3# self-attention
+ e4# 离散结构的链接
+ e5# 多个不同起始位置水平层+纵向综合层
+ e6# 跨层残差连接

## f# 决策层

+ f1# 取前n条分数较高的路径做Ens
+ f2# 记录多次CRF结果做Ens
+ f3# loss中加入对g与x有关的惩罚
+ f4# CRF中加入x做输入
+ f5# 取消B\I标签
+ f6# 在決策層做多頭：CRF、LSTM
+ f7# 使用方式：维度拼接、对位相加、门控系数