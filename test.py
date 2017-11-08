import numpy as np
import matplotlib.pyplot as plt

plt.figure()
# 通过栅格的形式创建布局方式,(3,3)创建3x3的布局形式，(0,0)绘制的位置，0行0列的位置绘制
# colspan:表示跨几列 rowspan:表示跨几行
ax1 = plt.subplot2grid((1,2),(0,0))
# 在ax1图中绘制一条坐标(1,1)到坐标(2,2)的线段
ax1.plot([1, 2], [1, 2])
# 设置ax1的标题  现在xlim、ylim、xlabel、ylabel等所有属性现在只能通过set_属性名的方法设置
ax1.set_title('ax1_title')  # 设置小图的标题
ax1.grid(True)
ax1.set_xlabel('hah')

ax2 = plt.subplot2grid((1,2),(0,1))

# 给对应的图绘制内容，这里只给ax4图绘制，属性通过set_xxx的模式设置

plt.show()