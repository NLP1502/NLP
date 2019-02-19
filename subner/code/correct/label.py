import difflib
# 句子标注脚本
# load1 错误的句子 一行一句
# load2 正确的句子 一行一句
# 生成文件:
# 每个错误一行，每个句子有一个空行
if __name__=="__main__":
    hd = difflib.HtmlDiff()
    loads = ''
    with open('C:/Users/asus/Desktop/NLP纠错/数据集/CGED2016/train_1_wrong.txt', 'r', encoding='gbk') as load1:
        loads = load1.readlines()
        load1.close()
    mem = ''
    with open('C:/Users/asus/Desktop/NLP纠错/数据集/CGED2016/train_1_true.txt', 'r', encoding='gbk') as load2:
        mem = load2.readlines()
        load2.close()
    with open('C:/Users/asus/Desktop/diff.txt', 'w', encoding='utf8') as fo:
        co = 0
        for i in loads:
            j = mem[co]
            co += 1
            s = difflib.SequenceMatcher(lambda x: x=="，", i, j)
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                if tag != "equal":
                    # print("%7s a[%d:%d] (%s) b[%d:%d] (%s)"
                    #       % (tag, i1, i2, i[i1:i2], j1, j2, j[j1:j2]))
                    if tag == "delete":
                        fo.write("R"+" "+str(i1+1)+" "+str(i2)+"\n")
                    elif tag == "insert":
                        fo.write("M" + " " + str(i1+1) + " " + str(i2+1)+"\n")
                    else:
                        fo.write("S" + " " + str(i1+1) + " " + str(i2) + "\n")
            fo.write("\n")
        print("out!")
        fo.close()