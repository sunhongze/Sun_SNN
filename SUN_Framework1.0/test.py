import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 150, 30)
y = [0.34,0.45,0.57,0.63,0.67,0.70,0.71,0.72,0.75,0.75,0.76,0.78,0.80,0.79,0.81,0.81,0.81,0.82,0.83,0.84,0.83,0.84,0.86,0.87,0.86,0.87,0.87,0.89,0.89,0.89]      # 曲线 y1

plt.figure()    # 定义一个图像窗口
plt.plot(x, y) # 绘制曲线 y1

plt.show()