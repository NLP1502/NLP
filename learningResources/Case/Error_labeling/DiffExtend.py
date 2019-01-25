# -*- coding: utf-8
from difflib import SequenceMatcher
from jieba import posseg, tokenize
import re


class DiffExtend:
    def __init__(self):
        self.initializer = ["", -5, -5, -5, -5]  # 操作和下标的初始化值
        self.INSERT = "insert"
        self.DELETE = "delete"
        self.REPLACE = "replace"
        self.DISORDER = "disorder"

    def label(source, target):
        de = DiffExtend()
        return de._label(source, target)

    # 获取指定index的分词段序号
    def _getIndex(self, seg_list, index):
        res = 0
        for seg in seg_list:
            if index < seg[1]:
                break
            res += 1
        return res

    # 考虑插入操作是否存在以下情况：
    # eg1 生效果 >> 产生效果； 原始：插入产； 修正：生替换为产生
    # eg2 癌症导致死 >> 癌症导致死亡； 原始：插入亡； 修正：死替换为死亡
    def _insert_inspect(self, operation, source, target, seg_list_s, seg_list_t, cur_seg):
        cur_operat, cur_is, cur_js, cur_it, cur_jt = operation
        tmp = seg_list_t[self._getIndex(seg_list_t, cur_it)]

        # 如果插入的字可以与前后组成单词
        if (cur_it, cur_jt) != tmp:
            tmp = target[tmp[0]: tmp[1]]
            tmp_1 = target[cur_it: cur_jt] + source[seg_list_s[cur_seg][0]: seg_list_s[cur_seg][1]]  # 后组词
            if cur_seg != 0:
                tmp_2 = source[seg_list_s[cur_seg - 1][0]: seg_list_s[cur_seg - 1][1]] + target[cur_it: cur_jt]  # 前组词
            else:
                tmp_2 = ""
            pos_1 = [p for _, p in posseg.cut(tmp_1)]
            pos_2 = [p for _, p in posseg.cut(tmp_2)]
            pos_1 = pos_1[0] if pos_1 else ""  # 后组词词性
            pos_2 = pos_2[0] if pos_2 else ""  # 前组词词性

            # 排除一下情况：
            # eg1 一微笑 >> 一个微笑； 正确：插入个； 修正：一替换为一个
            # eg2 个微笑 >> 一个微笑： 正确：插入一； 修正：个替换为一个
            if tmp_1 == tmp and pos_1 != 'm':
                res_operat, res_i, res_j = self.REPLACE, seg_list_s[cur_seg][0] + 1, seg_list_s[cur_seg][1]
            elif tmp_2 == tmp and pos_2 != 'm':
                res_operat, res_i, res_j = self.REPLACE, seg_list_s[cur_seg - 1][0] + 1, seg_list_s[cur_seg - 1][1]
            else:
                res_operat, res_i, res_j = cur_operat, cur_is + 1, cur_js + 1 if cur_is == cur_js else cur_js
        else:
            res_operat, res_i, res_j = cur_operat, cur_is + 1, cur_js + 1
        return res_operat, res_i, res_j

    def _replace_inspect(self, source, operation, seg_list_s, seg_list_t, cur_seg, extra_insert):
        cur_operat, cur_is, cur_js, cur_it, cur_jt = operation
        # 获取当前操作位置的单词的词性
        pos = [p for _, p in posseg.cut(source[seg_list_s[cur_seg][0]: seg_list_s[cur_seg][1]])]
        pos = pos[0] if pos else ""
        # 如果当前操作为替换，且替换的不是整个单词，且替换位置的单词不为数量词，则替换整个词
        # 这步修正是为了考虑以下情况：
        # eg 生产 >> 生活； 原始：产替换为活； 修正：生产替换为生活
        # 判断词性是为了避免以下情况：
        # eg 一个 >> 一种； 正确：个替换为种； 修正：一个替换为一种
        if (cur_is, cur_js) != seg_list_s[cur_seg] and pos != 'm':
            res_operat, res_i, res_j = self.REPLACE, seg_list_s[cur_seg][0] + 1, seg_list_s[cur_seg][1]
        else:
            res_operat, res_i, res_j = cur_operat, cur_is + 1, cur_js + 1 if cur_is == cur_js else cur_js

        # 如果替换的目标短语不是一个单词，则添加一步插入
        # eg 生活 >> 生产劳动； 原始：生活替换为生产劳动； 修正：生活替换为生产，插入劳动
        # 下面这句话的意思是替换的目标短语的起始位置和结束位置不在同一分词段
        if self._getIndex(seg_list_t, cur_it) != self._getIndex(seg_list_t, cur_jt - 1):
            extra_insert = True
        return res_operat, res_i, res_j, extra_insert

    def _label(self, source, target):
        # diff工具原理为子序列的最大匹配，若句中有多个匹配会导致混乱
        # 使用逗号分割得到短句后再将短句的match进行归并，默认短句不存在多个匹配
        source_list = source.split('，')
        target_list = target.split('，')
        if len(target_list) != len(source_list):
            match = SequenceMatcher(None, list(source), list(target)).get_opcodes()
        else:
            match = []
            len_1 = len_2 = 0
            # 遍历每一个正确错误的短句对
            for s_1, s_2 in zip(source_list, target_list):
                op_tmp = SequenceMatcher(None, list(s_1), list(s_2)).get_opcodes()  # 获取将wrong改正为correct的操作

                # diff的SequenceMatcher.get_opcodes()包含equal,replace,insert和delete四种
                # 显然我们不需要equal,但需要从insert和delete中摘取出disorder错误

                # 归并短句的操作以得到整句的操作，注意更新操作起止下标
                match.extend([[op[0], op[1] + len_1, op[2] + len_1, op[3] + len_2, op[4] + len_2] for op in op_tmp])
                len_1 += len(s_1) + 1
                len_2 += len(s_2) + 1
        match = [m for m in match if m[0] != "equal"]  # 去除equal操作

        # 获取两个句子的分词结果，格式为每一个分词段左闭右开区间的元组组成的列表
        # eg [(0, 1), (1, 4), (4, 8), (8, 9)]
        seg_list_s = [seg[1:] for seg in tokenize(source)]
        seg_list_t = [seg[1:] for seg in tokenize(target)]

        # 在分词序列末尾加入一个分词段作为padding，防止末尾插入时下标溢出
        seg_list_s.append((len(source), len(source) + 1))
        seg_list_t.append((len(target), len(target) + 1))
        k = 0
        result = []
        while k < len(match):
            cur_operat, cur_is, cur_js, cur_it, cur_jt = match[k]  # 当前修改操作
            nxt_operat, nxt_is, nxt_js, nxt_it, nxt_jt = match[k + 1] \
                if k < len(match) - 1 else self.initializer  # 下一步修改操作
            res_operat, res_i, res_j = self.initializer[:3]  # 修正后的修改操作结果
            cur_seg = self._getIndex(seg_list_s, cur_is)  # 当前操作起始位置在source中的段号
            nxt_seg = self._getIndex(seg_list_s, nxt_is)  # 当前操作起始位置在target中的段号
            extra_insert = False  # 是否有额外的插入操作

            # 如果是一个插入接着一个删除
            if (cur_operat, nxt_operat) == (self.INSERT, self.DELETE):
                # 如果插入与删除在同一单词内，认定为替换
                if nxt_seg == cur_seg:
                    res_operat, res_i, res_j = self.REPLACE, cur_is + 1, nxt_js
                    k += 1
                # 如果插入与删除隔了若干单词，且插入与删除的内容相同，认定为换序
                elif nxt_seg - cur_seg >= 1 and target[cur_it: cur_jt] == source[nxt_is: nxt_js]:
                    res_operat, res_i, res_j = self.DISORDER, cur_is + 1, nxt_js
                    k += 1
                # 暂时认定为插入，进行后续认定
                else:
                    res_operat, res_i, res_j = self._insert_inspect(match[k], source, target, seg_list_s, seg_list_t, cur_seg)

            # 如果是一个删除接着一个插入
            elif (cur_operat, nxt_operat) == (self.DELETE, self.INSERT):
                # 如果删除和插入是相邻单词，认定为替换
                if nxt_seg - cur_seg == 1:
                    res_operat, res_i, res_j = self.REPLACE, cur_is + 1, nxt_js
                    k += 1
                # 如果删除和插入隔了若干单词，且插入与删除的内容相同，认定为换序
                elif nxt_seg - cur_seg >= 2 and target[nxt_it: nxt_jt] == source[cur_is: cur_js]:
                    res_operat, res_i, res_j = self.DISORDER, cur_is + 1, nxt_js
                    k += 1
                # 认定为删除
                else:
                    res_operat, res_i, res_j = cur_operat, cur_is + 1, cur_js

            # 如果是一个插入接着一个替换
            elif (cur_operat, nxt_operat) == (self.INSERT, self.REPLACE):
                # 如果插入和替换在同一单词且插入的词与被替换的词相同，认定为替换后插入
                # eg 产生在工作 >> 生产队在工作
                # eg 原始：在产前插入生，将生替换为产队
                # eg 修正：产生换序，插入队
                if nxt_seg == cur_seg and target[cur_it: cur_jt] == source[nxt_is: nxt_js]:
                    res_operat, res_i, res_j = self.REPLACE, cur_is + 1, nxt_js
                    cur_seg = self._getIndex(seg_list_s, nxt_is)
                    extra_insert = True
                    k += 1
                # 如果插入和替换的单词相邻且插入的词与被替换的词相同，认定为换序后插入
                # eg 工具好的解决问题 >> 好的工具能解决问题
                # eg 原始：在好前插入工具，将好的替换为工具能
                # eg 修正：工具好的换序，插入能
                elif nxt_seg - cur_seg == 1 and target[cur_it: cur_jt] == source[nxt_is: nxt_js]:
                    res_operat, res_i, res_j = self.DISORDER, cur_is + 1, nxt_js
                    cur_seg = self._getIndex(seg_list_s, nxt_is)
                    extra_insert = True
                    k += 1
                # 暂时认定为插入，进行后续认定
                else:
                    res_operat, res_i, res_j = self._insert_inspect(match[k], source, target, seg_list_s, seg_list_t, cur_seg)

            # 如果是一个替换接着一个删除
            elif (cur_operat, nxt_operat) == (self.REPLACE, self.DELETE):
                # 如果替换和删除在同一单词且替换的目标词与被删除的词相同，认定为删除后替换
                # eg 好产生队在工作 >> 生产队在工作
                # eg 原始：将好替换为生，删除生
                # eg 修正：删除好，将产生替换为生产
                if nxt_seg - cur_seg == 1 and target[cur_it: cur_jt] == source[nxt_is: nxt_js]:
                    res_operat, res_i, res_j = self.REPLACE, cur_js + 1, nxt_js
                    result.append([self.DELETE, cur_is + 1, cur_js])
                    k += 1
                # 如果替换和删除的单词相邻且替换的目标词与被删除的词相同，认定为删除后换序
                # eg 还工具好的能解决问题 >> 好的工具能解决问题
                # eg 原始：将还替换为好的，删除好的
                # eg 修正：删除还，将工具与好的换序
                elif nxt_seg - cur_seg == 2 and target[cur_it: cur_jt] == source[nxt_is: nxt_js]:
                    res_operat, res_i, res_j = self.DISORDER, cur_js + 1, nxt_js
                    result.append([self.DELETE, cur_is + 1, cur_js])
                    k += 1
                # 暂时认定为替换，进行后续认定
                else:
                    res_operat, res_i, res_j, extra_insert = self._replace_inspect(
                        source, match[k], seg_list_s, seg_list_t, cur_seg, extra_insert)

            # 不是以上四种情况之一
            else:
                # 暂时认定为替换，进行后续认定
                if cur_operat == self.REPLACE:
                    res_operat, res_i, res_j, extra_insert = self._replace_inspect(
                        source, match[k], seg_list_s, seg_list_t, cur_seg, extra_insert)
                # 暂时认定为插入，进行后续认定
                elif cur_operat == self.INSERT and (cur_it, cur_jt) != seg_list_t[self._getIndex(seg_list_t, cur_it)]:
                    res_operat, res_i, res_j = self._insert_inspect(match[k], source, target, seg_list_s, seg_list_t, cur_seg)

                # 其他的一般情况
                else:
                    res_operat, res_i, res_j = cur_operat, cur_is + 1, cur_js + 1 if cur_is == cur_js else cur_js
            result.append([res_operat, res_i, res_j])
            if extra_insert:
                result.append([self.INSERT, seg_list_s[cur_seg][1] + 1, seg_list_s[cur_seg][1] + 1])
            k += 1
        return result
