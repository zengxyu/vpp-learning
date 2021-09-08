import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x, y, z, v = (np.random.random((4, 100)) - 0.5) * 15
c = np.abs(v)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
cax = ax.scatter(x, y, z, v, s=50, c=c, cmap=cmhot)

plt.show()
# ax1.scatter3D(gzs, gys, gxs, gvs, s=10, c=gvs, cmap=cmhot)  # 绘制散点图
