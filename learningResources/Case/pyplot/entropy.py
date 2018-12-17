import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

plt.rcParams['figure.figsize'] = (10, 10)
plt.figure(1)  # 创建图表1

x = np.linspace(0.0001, 0.9999, 100)
y = 1.0 - x
e = -(x*np.log(x) + y*np.log(y))
plt.plot(x, e)

plt.show()

fig = plt.figure(2)
ax = fig.gca(projection='3d')
x = np.linspace(0.0001, 0.4999, 100)
y = np.linspace(0.0001, 0.4999, 100)
X, Y = np.meshgrid(x, y)
z = 1-X-Y
e = -(x*np.log(x) + y*np.log(y) + z*np.log(z))

# Plot the surface.
surf = ax.plot_surface(X, Y, e, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.3)
cset = ax.contour(X, Y, e, 20,stride=1, zdir='x', offset=0, cmap=cm.coolwarm)
cset = ax.contour(X, Y, e, 20, zdir='y', offset=0.5, cmap=cm.coolwarm)
cset = plt.contour(X, Y, e, 20, zdir='z', offset=0, cmap=cm.coolwarm)

# Customize the z axis.
# ax.set_zlim(0, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
