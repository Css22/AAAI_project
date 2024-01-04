
import numpy as np
import matplotlib.pyplot as plt
path = '/data/zbw/course/AAAI/project/work/processed_data/test/9898.npy'
data = np.load(path)
channel_to_visualize = 0  # 这里选择第一个通道
channel_data = data[channel_to_visualize]
plt.imshow(channel_data, cmap='gray')  
# plt.colorbar()  # 添加颜色条（可选）
plt.show()