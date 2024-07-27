import torch
import os
import numpy as np

_all_dataset_root = 'data/SceneVerse'
_openscene_root = 'data/SceneVerse/OpenScene_Scan_Features'

def _load_scan(pcd_path, inst2label_path, scan_name):
    pcd_data = torch.load(os.path.join(pcd_path, f'{scan_name}'))
    try:
        inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}"))
    except:
        inst_to_label = None
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]

    pcds = np.concatenate([points, colors], 1)
    return points, colors, pcds, instance_labels, inst_to_label

def _load_scan_data(scan_name, dataset_name='HM3D'):
    dataset_root = os.path.join(_all_dataset_root, dataset_name)
    scan_data_root = os.path.join(dataset_root, 'scan_data')
    inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
    inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}")) 
    dataset_root = os.path.join(_openscene_root, dataset_name)
    dict = torch.load(os.path.join(dataset_root, scan_name), map_location='cpu')
    points = dict['points'].numpy().astype(np.float32)
    colors = dict['colors'].numpy()
    features = dict['features'].numpy().astype(np.float32)
    instance_labels = dict['instance_labels'].numpy()
    
    return points, colors, features, instance_labels, inst_to_label

def down_sample(points, colors, instance_labels=None, featrues=None, npoint=None):
    pcd_idxs = np.random.choice(len(points), size=npoint, replace=len(points) < npoint)
    points = points[pcd_idxs]
    colors = colors[pcd_idxs]
    instance_labels = instance_labels[pcd_idxs] if not instance_labels is None else None
    featrues = featrues[pcd_idxs] if not featrues is None else None
    return points, colors, instance_labels, featrues



dataset_name = 'HM3D'
scan_name = '00591-JptJPosx1Z6'
instance_room_id = 'sub004'
region_id = 'sub004-sub008-sub012'

room_center = torch.load(os.path.join('data/SceneVerse/HM3D/scan_data/room_center', '{}.pth'.format(scan_name)))
        
points = []
colors = []
features = []
instance_labels = []
inst_to_label = {}
for ri, room_id in enumerate(region_id.split('-')):
    pts, cols, fts, ils, itl = _load_scan_data(f'{scan_name}_{room_id}.pth', dataset_name)
    points.extend(pts + room_center[room_id]['center'])
    colors.extend(cols)
    features.extend(fts)
    if room_id == instance_room_id:
        instance_labels.extend(ils)
    else:
        instance_labels.extend(np.zeros_like(ils).astype(ils.dtype))
    inst_to_label[room_id] = itl

points = np.array(points)
colors = np.array(colors)
features = np.array(features)
instance_labels = np.array(instance_labels)


# Get HD Info
N = 240000
hd_points, hd_features, hd_colors, hd_instance_labels = down_sample(points, features, colors, instance_labels, npoint=N)

# remove top 10% points and colors
top10 = np.percentile(hd_points[:,2], 70)
mask = hd_points[:,2] < top10
hd_points = hd_points[mask]
hd_colors = hd_colors[mask]


# visiualize
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(hd_points)
pcd.colors = o3d.utility.Vector3dVector(hd_colors/255)
o3d.visualization.draw_geometries([pcd])