#coding:utf8
import math
def get_score_topk(predict_label_and_marked_label_list, visible_top_num, num_classes):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  #总命中标签数量
    right_label_at_pos_num = [0 for _ in range(visible_top_num)]  #在各个位置上总命中数量
    sample_num = 0   #总问题数量
    total_classes_num = [0 for _ in range(num_classes)]
    right_classes_num = [0 for _ in range(num_classes)]
    precision_classes_num = [0 for _ in range(num_classes)]
    all_marked_label_num = 0    #总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        for _ in marked_labels:
            total_classes_num[int(_)] += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:     #命中
                right_classes_num[int(label)] += 1
                right_label_num += 1
                right_label_at_pos_num[pos] += 1
    for i_, r_, t_ in zip(range(num_classes), right_classes_num, total_classes_num):
        if t_ != 0:
            precision_classes_num[i_] = r_ / float(t_)
    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(3.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    return (precision * recall) / (precision + recall +0.0000000000001),precision,recall,right_label_at_pos_num,precision_classes_num

def get_score(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]
    
    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  #总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  #在各个位置上总命中数量
    sample_num = 0   #总问题数量
    all_marked_label_num = 0    #总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:     #命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(3.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    return (precision * recall) / (precision + recall +0.0000000000001),precision,recall,right_label_at_pos_num

if __name__ == '__main__':
    import numpy as np

    predict_label_and_marked_label_list = [([4,9,2,3], [1,2,3,4,5]),
     ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    a, b, c, d, e = get_score_topk(predict_label_and_marked_label_list, 20,10)
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)