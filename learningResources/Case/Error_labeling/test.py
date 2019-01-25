from DiffExtend import DiffExtend
import re

xml_file_name = "2016_CGED_TrainingSet.txt"  # 存储句对和正确标签的xml文件
# xml_file_name = "train.release.xml"  # 存储句对和正确标签的xml文件
out_file_name = "label_result.txt"  # 存储脚本标注的标签的输出文件


# 从xml文件获取句对数据
def getData(filename):
    with open(filename, 'r', encoding='utf-8') as in_file:
        data = in_file.read()
        wrong = re.findall(re.compile(r'<TEXT id="\w+">[\n](.*)[\n]</TEXT>'), data)  # 正则匹配错误句子
        correct = re.findall(re.compile(r'<CORRECTION>[\n](.*)[\n]</CORRECTION>'), data)  # 正则匹配正确句子
    return correct, wrong


# 将四种操作变换成对应的标记
def transOpt(operation):
    if operation == "delete":  # 刪除
        return "R"
    if operation == "replace":  # 替换
        return "S"
    if operation == "insert":  # 插入
        return "M"
    if operation == "disorder":  # 换序
        return "W"


# 获取正确的句子和错误的句子
correct, wrong = getData(xml_file_name)
with open(out_file_name, 'w', encoding='utf-8') as out:
    # 遍历每一个错误和正确句子的句对
    # 默认将标签打在wrong上，若wrong与correct交换，则标签打在correct上
    for correct, wrong in zip(correct, wrong):
        operations = DiffExtend.label(source=wrong, target=correct)
        for opt in operations:
            out.write(transOpt(opt[0]) + " " + str(opt[1]) + " " + str(opt[2]) + '\n')
        out.write('\n')


# 制作正确的修改标签文件data.txt以供evaluate使用
with open("data.txt", 'w', encoding='utf-8') as out:
    with open(xml_file_name, 'r', encoding='utf-8') as in_file:
        data = in_file.read()
        docs = re.findall(re.compile(r'<DOC>((.*[\n])+?)</DOC>'), data)  # 正则表达式匹配一个文档
        k = 0
        for doc in docs:
            # 正则表达式匹配文档内的纠错信息
            infos = re.findall(re.compile(r'<ERROR start_off="([0-9]+)" end_off="([0-9]+)" type="([SRWM])">'), doc[0])
            for info in infos:
                out.write(info[2] + " " + info[0] + " " + info[1] + "\n")
            out.write('\n')
            k += 1