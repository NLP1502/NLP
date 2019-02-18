import math
rt = [1,2,3]
pt = [1,3,7,99]
rtn = len(rt)
ptrn = 0
p1 = 0
p2 = 0
for i in range(len(pt)):
    p2 += 1/math.log(i+3)
    if pt[i] in rt:
        ptrn += 1
        p1 += 1 / math.log(i + 3)
p = p1 / p2
r = ptrn / rtn
f = 2*p*r / (p+r)
print(f)

