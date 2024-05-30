import open3d
import clip
import os
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cuda', jit=False)

def encode_text(prompt):
    text = clip.tokenize([prompt])
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

dataset = 'HM3D'
scan_name = '00723-hWDDQnSDMXb_sub023'
feature_dir = 'data/SceneVerse/OpenScene_Scan_Features/{}'.format(dataset)

xyz = torch.load(os.path.join(feature_dir, '{}_xyz.pt'.format(scan_name)), map_location='cpu')
color = torch.load(os.path.join(feature_dir, '{}_color.pt'.format(scan_name)), map_location='cpu')
features = np.load(os.path.join(feature_dir, '{}_distill_fts.npy'.format(scan_name)))
features = torch.from_numpy(features).cuda().half()
features = features / features.norm(dim=-1, keepdim=True)

prompt = 'toilet'
text_features = encode_text(prompt)

# calculate similarity
similarity = (features @ text_features.T).squeeze().detach().cpu().numpy()
# softmax
# similarity = np.exp(similarity) / np.exp(similarity).sum()
similarity = (similarity - np.min(similarity)) / (np.max(similarity) - np.min(similarity))
print(np.max(similarity))
print(np.min(similarity))

# draw heatmap on point clouds
map_colors = ["blue", "yellow", "red"]  # 蓝色到红色
cmap = LinearSegmentedColormap.from_list("mycmap", map_colors)
activation_colors = cmap(similarity)
## normalization

mixed_colors = 0.5 * (color/255) + 0.5 * activation_colors[:,:3]

# visualize
print(xyz.shape)
pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(xyz)
pcd.colors = open3d.utility.Vector3dVector(mixed_colors)
open3d.visualization.draw_geometries([pcd])

