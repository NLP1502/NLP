# ax.plot_wireframe(X, Y, Z, *args, **kwargs)
# X,Y,Z：输入数据
# rstride:行步长
# cstride:列步长
# rcount:行数上限
# ccount:列数上限
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

X = np.array([[1,10,20]])
Y = np.array([[1,10,20]])
Z = np.array([[1,10,-10]])
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color='r')

plt.show()
