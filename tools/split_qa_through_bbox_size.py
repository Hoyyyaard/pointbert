import json
import os 
import random
import numpy as np
from tqdm import tqdm
import torch

def visualization(data_list, save_path): 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(data_list, bins=10, edgecolor='black')
    plt.title('Histogram of Random Numbers')
    plt.grid(True)
    plt.savefig(save_path)

def load_scan(pcd_path, inst2label_path, scan_name):
    pcd_data = torch.load(os.path.join(pcd_path, f'{scan_name}'))
    try:
        inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}"))
    except:
        inst_to_label = None
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]

    pcds = np.concatenate([points, colors], 1)
    return points, colors, pcds, instance_labels, inst_to_label

def load_scan_data(scan_name, dataset_name):
    dataset_root = os.path.join('data/SceneVerse', dataset_name)
    annotation_root = os.path.join(dataset_root, 'annotations')
    scan_data_root = os.path.join(dataset_root, 'scan_data')
    inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
    pcd_path = os.path.join(scan_data_root,'pcd_with_global_alignment')
    points, colors, pcds, instance_labels, inst_to_label = load_scan(pcd_path, inst2label_path, scan_name)

    return points, colors, instance_labels, inst_to_label

os.makedirs('data/SceneVerse/meta_data', exist_ok=True)

# ------------------------------------------------ Scan qa -----------------------------------------
all_scene_qa = []
scanqa_anno = json.load(open(f'data/SceneVerse/ScanNet/annotations/qa/ScanQA_v1.0_val.json'))
for si,scene_cap in enumerate(scanqa_anno):
    scan_name = scene_cap['scene_id']
    all_scene_qa.append({'dataset_name':'ScanNet', 
                                "scan_name":scan_name, 
                                "anno":scene_cap, 
                                "task_name": "scene_qa",
                                'episode_id':'{}#{}#{}'.format('ScanNet', scan_name, scene_cap['question'])
                                })

BBOX_THRES = 0.05 # ~ keyboard size
area_sizes = {}
small_unique_epi_ids = []
for scene_qa in tqdm(all_scene_qa):
    dataset_name = scene_qa['dataset_name']
    scan_name = scene_qa['scan_name']
    anno = scene_qa['anno']
    points, colors, instance_labels, inst_to_label = load_scan_data(f'{scan_name}.pth', dataset_name)
    scene_qa['points'] = points
    scene_qa['colors'] = colors
    scene_qa['instance_labels'] = instance_labels
    scene_qa['inst_to_label'] = inst_to_label

    target_obj_id = random.choice(anno['object_ids'])
    object_points = points[instance_labels == target_obj_id]
    object_sizes = np.max(object_points, 0) - np.min(object_points, 0)
    object_sizes = object_sizes[0] * object_sizes[1] * object_sizes[2]
    if object_sizes < BBOX_THRES:
        small_unique_epi_ids.append(scene_qa['episode_id'])
        # print('{} size:{}'.format(inst_to_label[target_obj_id], object_sizes))
        
    area_size = np.max(points, 0) - np.min(points, 0)
    area_size = area_size[0] * area_size[1]
    area_sizes[scan_name] = area_size.tolist()

with open('scanqa.json', 'w') as f:
    json.dump(area_sizes, f)

area_sizes = list(area_sizes.values())
scannet_avg_area_size = sum(area_sizes)/len(area_sizes)
print('small_unique_epi_ids:', len(small_unique_epi_ids))
print('Scannet scan average area size: {}'.format(scannet_avg_area_size))
# with open('data/SceneVerse/meta_data/ScanQA_Bbox_Small_Than-{}_Val.json'.format(BBOX_THRES), 'w') as f:
#     json.dump(small_unique_epi_ids, f)

# visualization(area_sizes, "scanqa.png")
    
    
# ------------------------------------------------ HD hm3d qa -----------------------------------------
qa_source_dir_f = 'data/SceneVerse/HM3D/annotations/qa/qa_pairs/val.json'
with open(qa_source_dir_f, 'r') as f:
    datas = json.load(f)
all_scene_qa = []
for scan_name, episodes in datas.items():
    for epi in episodes:
        for qa in epi['qa_pairs']:
            anno = {
                'question': qa['question'],
                'answers': [qa['answer']],
                'type': qa['type'],
                'target_id': epi['target_id'],
                'instance_type': epi['instance_type'],
                # 'utterance': epi['utterance'],
            }
            all_scene_qa.append({'dataset_name':'HM3D', 
                                    "scan_name":scan_name, 
                                    'instance_room_id': epi['scan_id'].split('_')[-1],
                                    "anno":anno, 
                                    "task_name": "scene_qa",
                                    'region_id': qa['region_id'],
                                    'episode_id':'{}#{}#{}#{}'.format('HM3D', scan_name, qa['region_id'], qa['question'])
                                    })
small_unique_epi_ids = []
area_sizes = {}
large_than_scannet_num = 0
BBOX_THRES = 0.01 # ~ keyboard size
for data in tqdm(all_scene_qa):
    dataset_name, scan_name, anno, task_name, region_id = data['dataset_name'], data['scan_name'], data['anno'], data['task_name'], data['region_id']
    instance_room_id = data['instance_room_id']
    tgt_id = int(anno['target_id'])
    room_center = torch.load(os.path.join('data/SceneVerse/HM3D/scan_data/room_center', '{}.pth'.format(scan_name)))
    anno = data['anno']
    
    points = []
    colors = []
    instance_labels = []
    inst_to_label = {}
    for room_id in region_id.split('-'):
        pts, cols, ils, itl = load_scan_data(f'{scan_name}_{room_id}.pth', dataset_name)
        points.extend(pts + room_center[room_id]['center'])
        colors.extend(cols)
        if room_id == instance_room_id:
            instance_labels.extend(ils)
        else:
            instance_labels.extend(np.zeros_like(ils).astype(ils.dtype))
        inst_to_label[room_id] = itl
    
    points = np.array(points)
    colors = np.array(colors)
    instance_labels = np.array(instance_labels).astype(np.int64)
    
    target_obj_id = int(anno['target_id'])
    object_points = points[instance_labels == target_obj_id]
    object_sizes = np.max(object_points, 0) - np.min(object_points, 0)
    object_sizes = object_sizes[0] * object_sizes[1] * object_sizes[2]
    if object_sizes < BBOX_THRES:
        small_unique_epi_ids.append(data['episode_id'])
        # print('{} size:{}'.format(anno['instance_type'], object_sizes))
    area_size = np.max(points, 0) - np.min(points, 0)
    area_size = area_size[0] * area_size[1]
    area_sizes[region_id] = area_size.tolist()

with open('hm3dxrqa.json', 'w') as f:
    json.dump(area_sizes, f)

area_sizes = list(area_sizes.values())
print('Hm3d scan average area size: {}'.format(sum(area_sizes)/len(area_sizes)))
print('small_unique_epi_ids:', len(small_unique_epi_ids))
# with open('data/SceneVerse/meta_data/HDHm3dQA_Bbox_Small_Than-{}_Val.json'.format(BBOX_THRES), 'w') as f:
#     json.dump(small_unique_epi_ids, f)
# visualization(area_sizes, "hm3dqa.png")