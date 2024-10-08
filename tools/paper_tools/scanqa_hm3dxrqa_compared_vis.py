import seaborn as sns
import json
import numpy as np
import matplotlib.pyplot as plt

scanqa_areas = json.load(open('data/SceneVerse/meta_data/scanqa.json'))
hm3dxrqa_areas = json.load(open('data/SceneVerse/meta_data/hm3dxrqa.json'))
hm3dxrqa_areas = {k:v for k,v in hm3dxrqa_areas.items() if not int(v) > 200}

scanqa_areas = list(scanqa_areas.values())
hm3dxrqa_areas = list(hm3dxrqa_areas.values())

# Create a cumulative count of scenes based on area sizes
scanqa_counts, scanqa_bins = np.histogram(scanqa_areas, bins=50)
hm3dxrqa_counts, hm3dxrqa_bins = np.histogram(hm3dxrqa_areas, bins=50)

# Get the mid points of bins for smooth curve
scanqa_bin_mids = (scanqa_bins[:-1] + scanqa_bins[1:]) / 2
hm3dxrqa_bin_mids = (hm3dxrqa_bins[:-1] + hm3dxrqa_bins[1:]) / 2

# smooth
# N = 1
# scanqa_bin_mids = scanqa_bin_mids[::N]
# scanqa_counts = scanqa_counts[::N]
# hm3dxrqa_bin_mids = hm3dxrqa_bin_mids[::N]
# hm3dxrqa_counts = hm3dxrqa_counts[::N]

# Plot the data using smooth curves
plt.figure(figsize=(10, 6))
# sns.lineplot(x=scanqa_bin_mids, y=scanqa_counts, label='ScanQA', marker='o')
# sns.lineplot(x=hm3dxrqa_bin_mids, y=hm3dxrqa_counts, label='HM3D-XR-QA', marker='x')
sns.kdeplot(scanqa_areas, label='ScanQA', fill=True)
sns.kdeplot(hm3dxrqa_areas, label='HM3D-XR-QA', fill=True)


# Add labels and title
plt.xlabel('Area Size (m²)', fontsize=22)
plt.ylabel('Number of Scenes', fontsize=22)
plt.title('Comparison of Scene Areas Between ScanQA and HM3D-XRQA', fontsize=22)
plt.legend(fontsize=22)  # 设置图例字体大小
plt.xticks(fontsize=22)
plt.yticks([])  # 移除 y 轴坐标
# Show the plot
plt.tight_layout()
# plt.show()
plt.savefig('data/SceneVerse/meta_data/scanqa_hm3dxrqa_compared_vis.pdf')
