import difflib

# 数据验证程序
# load1 标准数据集
# load2 待验证的数据集


if __name__ == "__main__":
    hd = difflib.HtmlDiff()
    # 标准数据集
    with open('data.txt', 'r', encoding='gbk') as load1:
        loads = load1.readlines()
        load1.close()
    # 待验证的数据集
    with open('diff.txt', 'r', encoding='gbk') as load2:
        mem = load2.readlines()
        load2.close()
    j = 0
    aS = [0, 0]
    aW = [0, 0]
    aM = [0, 0]
    aR = [0, 0]
    s1 = []
    s2 = []
    tr = 0
    al = 0
    i = 0
    while i < len(loads):
        print(i)
        print(j)
        S = [0, 0]
        R = [0, 0]
        M = [0, 0]
        W = [0, 0]
        al += 1
        while loads[i] != "\n":
            s1.append(loads[i])
            i += 1
        i += 1
        while mem[j] != "\n":
            s2.append(mem[j])
            j += 1
        j += 1
        for i1 in s1:
            for j1 in s2:
                if i1.split(" ")[0] == "R" and j1.split(" ")[0] == "R":
                    if int(i1.split(" ")[1]) - 1 == int(j1.split(" ")[1]) or int(i1.split(" ")[1]) + 1 == int(
                            j1.split(" ")[1]):
                        R[0] += 1
                if i1 == j1:
                    if i1.split(" ")[0] == "W":
                        W[0] += 1
                    elif i1.split(" ")[0] == "R":
                        R[0] += 1
                    elif i1.split(" ")[0] == "M":
                        M[0] += 1
                    else:
                        S[0] += 1
        for h in s1:
            if h.split(" ")[0] == "W":
                W[1] += 1
            elif h.split(" ")[0] == "R":
                R[1] += 1
            elif h.split(" ")[0] == "M":
                M[1] += 1
            else:
                S[1] += 1
        if W[1] == W[0] and R[1] == R[0] and M[1] == M[0] and S[0] == S[1]:
            tr += 1
        aW[0] += W[0]
        aW[1] += W[1]
        aR[0] += R[0]
        aR[1] += R[1]
        aS[0] += S[0]
        aS[1] += S[1]
        aM[0] += M[0]
        aM[1] += M[1]
        s1.clear()
        s2.clear()
        print("\n总句子数:" + str(al) + " 判错完全正确的句子数:" + str(tr) + "\n类型S判断正确" + str(aS[0]) + " 类型S验证" + str(
            aS[1]) + "\n类型R判断正确" + str(aR[0]) + " 类型R验证" + str(aR[1]) + "\n类型M判断正确" + str(aM[0]) + " 类型M验证" + str(
            aM[1]) + "\n类型W判断正确" + str(aW[0]) + " 类型W验证" + str(aW[1]))
