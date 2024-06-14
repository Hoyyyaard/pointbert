import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import cv2

def plot_filtered_point_cloud_with_colors(ptcloud, color, z_min, z_max):
    # 过滤点云数据，移除z轴范围外的点
    # mask = (ptcloud[:, 2] >= z_min) & (ptcloud[:, 2] <= z_max)
    # filtered_ptcloud = ptcloud[mask]
    filtered_ptcloud = ptcloud

    fig = plt.figure(figsize=(8, 8))
    
    # 提取过滤后的x, y, z数据
    x, y, z = filtered_ptcloud[:, 0], filtered_ptcloud[:, 1], filtered_ptcloud[:, 2]
    # x, z, y = filtered_ptcloud.transpose(1, 0)
    # 提取过滤后的颜色数据
    colors = color / 255.0  # 将颜色值缩放到[0, 1]范围内

    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.view_init(30, 45)
    
    # 获取过滤后的点云数据的最大值和最小值
    max_val, min_val = np.max(filtered_ptcloud[:, :3]), np.min(filtered_ptcloud[:, :3])
    ax.set_xbound(min_val, max_val)
    ax.set_ybound(min_val, max_val)
    ax.set_zbound(min_val, max_val)
    
    # 绘制过滤后的点云，使用RGB颜色
    # ax.scatter(x, y, z, c=colors, s=10)  # s是点的大小，可以根据需要调整
    ax.scatter(x, y, z, c=colors, cmap='jet')
    ax.view_init(elev=90, azim=0)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img

# 示例点云数据，包含RGB颜色
# 生成一些随机点云数据和对应的随机颜色
# ptcloud = np.random.rand(100, 6) * 10
# ptcloud[:, 3:] = np.random.rand(100, 3) * 255  # 随机颜色
data = torch.load("data/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment/scene0000_00.pth")
ptcloud = data[0]
color = data[1]

fig = plt.figure(figsize=(8, 8))
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax = fig.gca(projection=Axes3D.name, adjustable='box')
ax.axis('off')
ax.scatter3D(ptcloud[:, 0], ptcloud[:, 1], ptcloud[:, 2], zdir='z', c=color/255)
ax.view_init(elev=90, azim=0)
# max, min = np.max(ptcloud), np.min(ptcloud)
# ax.set_xbound(min, max)
# ax.set_ybound(min, max)
# ax.set_zbound(min, max)
# ax.subpl
# fig.subplots_adjust(left=0, right=1, top=2, bottom=-1)
ax.dist = 5.8
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig.canvas.draw()
img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))


fig, axs = plt.subplots(6, 6, figsize=(64, 64))
axs[0,0].axis('off')
axs[0,0].imshow(img)
fig.savefig("test.png")  
# reverse rgb to bgr
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('point_cloud.png', img)
 


# # 定义需要保留的z轴范围
# z_min = -10
# z_max = 10

# # 绘制并返回图像
# img = plot_filtered_point_cloud_with_colors(ptcloud, color, z_min, z_max)

# # 显示图像
# plt.imshow(img, aspect='auto')
# plt.axis('off')
# plt.show()
