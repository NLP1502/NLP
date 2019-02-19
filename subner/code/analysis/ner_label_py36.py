from functools import reduce
import csv

# 返回一个句子sentence list，其中，每一个sentence是一个word list
def get_sentence(file_url):
    with open(file_url,encoding='utf-8') as f:
        data = f.read().split('\n')
    sentences = []
    star = end =0
    for end in range(len(data)):
        if data[end] == '':
            sentences.append(data[star:end])
            star = end + 1
    return sentences


# 文件最后一列 1：预测正确，2：未预测出，3：预测错误
def ner_label(true_file_url, predict_file_url, window=2):
    true_sentences = get_sentence(true_file_url)
    predict_sentences = get_sentence(predict_file_url)
    label = ('B-M', 'B-R', 'B-W', 'B-S')
    result = []
    for i in range(len(true_sentences)):
        for j in range(len(true_sentences[i])):
            if true_sentences[i][j].endswith(label) or predict_sentences[i][j].endswith(label):
                flag = 0  # 代表实体标签在真实句子中
                if true_sentences[i][j].endswith(label):
                    sentence = true_sentences[i]
                else:
                    sentence = predict_sentences[i]
                    flag = 1    #代表实体标签在预测句子中
                word = sentence[j]
                lable_start = j
                word_lable = word.split('-')[1]
                lable_end = lable_start
                for lable_end in range(lable_start, len(sentence) - 1):
                    if sentence[lable_end + 1].split('\t')[1] != 'I-' + word_lable:
                        break
                pre_ok = bool(lable_start >= window)
                pos_ok = bool(lable_start + window <= len(sentence) - 1)
                if pre_ok and pos_ok:
                    str = reduce(lambda x, y: x.split('\t')[0] + y.split('\t')[0],
                                 sentence[lable_start - window: lable_end + window + 1],'')
                elif (not pre_ok) and pos_ok:
                    str = '%' * (window - lable_start) + reduce(lambda x, y: x.split('\t')[0] + y.split('\t')[0],
                                                             sentence[0: lable_end + window + 1])
                elif pre_ok and (not pos_ok):
                    str = reduce(lambda x, y: x.split('\t')[0] + y.split('\t')[0],
                                 sentence[lable_start - window: len(sentence)]) + '%' * (window + lable_end - len(sentence) + 1)
                else:
                    str = '%' * (window - lable_start) + \
                          reduce(lambda x, y: x.split('\t')[0] + y.split('\t')[0],
                                 sentence[0: len(sentence)]) + '%' * (window + lable_start - len(sentence) + 1)
                if flag == 0:
                    true_label = word_lable
                    temp = predict_sentences[i][j].split('\t')[1]
                    if temp == 'O':
                        predict_label = temp
                        state = 2
                    else:
                        predict_label = temp.split('-')[1]
                        if predict_label == true_label:
                            state = 1
                        else:   # 真实标签和预测标签的实体类型不同
                            state = 3
                else:
                    predict_label = word_lable
                    temp = true_sentences[i][j].split('\t')[1]
                    if temp == 'O':
                        true_label = temp
                        state = 3
                    else:
                        true_label = temp.split('-')[1]
                        if predict_label == true_label:
                            state = 1
                        else:  # 真实标签和预测标签的实体类型不同
                            state = 3
                result.append((str, true_label, predict_label, state))
    result_name = true_file_url.split('.')[0] + '_result.csv'
    with open(result_name, 'w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerows(result)


if __name__ == '__main__':
    ner_label('true_test.txt', 'predict_test.txt', window=0)    #window缺省为2

