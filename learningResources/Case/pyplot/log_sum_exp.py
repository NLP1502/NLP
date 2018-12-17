# log_sum_exp是一个额外提高量，当某元素与最大
# 值相差0时带来0.69的提升，相差1时为0.31，多
# 个元素的提升量叠加时效果衰减。该方法做loss
# 时倾向于获得与再大值跟接近的其他非最大值。
#
import numpy as np
import matplotlib.pyplot as plt

def argmax(vec):
    idx = np.max(vec)
    return idx

plt.rcParams['figure.figsize'] = (5, 5)
plt.figure(1)
x = [0,1]
vec = np.array([2,2])
plt.plot(x, vec, label = 'vec')
max_score = argmax(vec)
max_score_broadcast = max_score.repeat([vec.size])
plt.plot(x, np.log(max_score_broadcast), label = 'log_max')
exp = np.exp(vec - max_score_broadcast)
plt.plot(x, exp, label = 'exp_dif')
y = np.log(np.sum(np.exp(vec - max_score_broadcast)))
print(np.sum(np.exp(vec - max_score_broadcast)))
print(y)
y2 = y.repeat([vec.size])
plt.plot(x, y2, label = 'log_sum_exp_dif')
plt.legend()
plt.show()