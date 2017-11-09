import numpy as np
import matplotlib.pyplot as plt
import time
import math
plt.figure()
for i in range(10000):
    plt.scatter(i,math.sin(i))
    plt.savefig('1.png')
    time.sleep(0.1)
# 给对应的图绘制内容，这里只给ax4图绘制，属性通过set_xxx的模式设置
# plt.show()