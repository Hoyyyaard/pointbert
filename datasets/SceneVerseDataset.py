import numpy as np
import os, sys
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *
from einops import rearrange
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pandas as pd
import glob
from scipy.spatial import cKDTree
from tqdm import tqdm
import random
import copy
import json
from transformers import AutoTokenizer


@DATASETS.register_module()
class Objaverse(Dataset):

    def __init__(self, num_points, split):
        super().__init__()
        self.split = split
        self._npoint = num_points
        self.cap3d_root ='data/LEO_DATA/Cap3d_root/Cap3D_pcs_pt'
        self.create_obj_cap_dict()
        # self.obj_ids = self.obj_ids[:100000]
        # if split == 'train':
        #     self.obj_ids = self.obj_ids[:-1000]
        # else:
        #     self.obj_ids = self.obj_ids[-1000:]

    def create_obj_cap_dict(self):
        obj_csv = pd.read_csv(os.path.join(self.cap3d_root, 'Cap3D_automated_Objaverse_no3Dword.csv'), header=None)
        self.obj_ids = []
        self.obj_cap_dict = {}
        for obj_id, cap in zip(obj_csv[0].values, obj_csv[1].values):
            # remove redundant quotation marks, here we do not directly strip because the mark may appear only at one side
            if cap.startswith('"') and cap.endswith('"'):
                cap = cap.strip('"')
            elif cap.startswith("'") and cap.endswith("'"):
                cap = cap.strip("'")

            self.obj_ids.append(obj_id)
            self.obj_cap_dict[obj_id] = cap

    def load_obj_pcd(self, obj_id):
        pcd = torch.load(
            os.path.join(self.cap3d_root, f'Cap3D_pcs_pt/{obj_id}.pt'),
            map_location='cpu'
        )   # (6, 16384)
        pcd = rearrange(pcd, 'c n -> n c')   # (16384, 6), xyz (m) + rgb (uint8)
        # pcd[:, 3:] = pcd[:, 3:] / 127.5 - 1   # (16384, 6), xyz (m) + rgb (float, [-1, 1])
        return pcd.numpy()
    
    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, index):
        obj_id = self.obj_ids[index]
        obj_pcd = self.load_obj_pcd(obj_id)
        return obj_pcd[:, :3], obj_pcd[:, 3:]
        
        
@DATASETS.register_module()
class SceneVerseDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        
        # Load scene level data
        self._npoint = config.N_POINTS
        self._group_size = config.GROUP_SIZE
        self._num_groups = config.NUM_GROUP
        _all_dataset_name = ['3RScan', 'HM3D', 'MultiScan', 'ARKitScenes', 'ScanNet']
    
        self._all_dataset_root = 'data/SceneVerse'
        self._all_scans = []
        self.dataset_names = []
        for dataset_name in _all_dataset_name:
            path = f'{self._all_dataset_root}/{dataset_name}/scan_data/pcd_with_global_alignment'
            if not os.path.exists(path):
                continue
            data = glob.glob(f'{path}/*.pth')
            data = [d.split('/')[-1] for d in data]
            self.dataset_names.extend([dataset_name] * len(data))
            self._all_scans.extend(data)
            print_log(f'[DATASET] {len(data)} scans from {dataset_name} were loaded', logger = 'SceneVerse')
        self._all_scans_datasets = []
        for dn, sc in zip(self.dataset_names, self._all_scans):
            self._all_scans_datasets.append((dn, sc))
        
        if not config.PREPROCESS:
            # Load region level data
            self._load_aug_region_data()
            self._region_npoint = config.REGION_N_POINTS
            self._region_group_size = config.REGION_GROUP_SIZE
            self._region_num_groups = config.REGION_NUM_GROUP
            
            # Load instance level data
            self._instance_npoint = config.INSTANCE_N_POINTS
            self._instance_group_size = config.INSTANCE_GROUP_SIZE
            self._instance_num_groups = config.INSTANCE_NUM_GROUP
            self._load_objaverse_data()
            
            if config.subset == 'train':
                self._all_scans_datasets = self._all_scans_datasets[:-1000]
                self._all_region = self._all_region[:-10000]
                self.objaverse_data.obj_ids = self.objaverse_data.obj_ids[:200000]
            else:
                self._all_scans_datasets = self._all_scans_datasets[-1000:]
                self._all_region = self._all_region[-10000:]
                self.objaverse_data.obj_ids = self.objaverse_data.obj_ids[-10000:]           

            # As diffent dataset has different number of points, we need to specify the dataset squence order 
            # to make sure samples from on batch come from the same level dataset
            batch_size_pre_rank = config.tbs
            self.order_episodes = []
            self.order_levels = []
            random.shuffle(self._all_scans_datasets)
            random.shuffle(self._all_region)
            random.shuffle(self.objaverse_data.obj_ids)
            while self._all_region or self._all_scans_datasets or self.objaverse_data.obj_ids:
                if self._all_scans_datasets:
                    if len(self._all_scans_datasets) < batch_size_pre_rank:
                        pop_num  = len(self._all_scans_datasets)
                        [self._all_scans_datasets.pop(0) for _ in range(pop_num)]
                    else:
                        self.order_episodes.extend([self._all_scans_datasets.pop(0) for _ in range(batch_size_pre_rank)])
                        self.order_levels.extend(['scene'] * batch_size_pre_rank)
                if self._all_region:
                    if len(self._all_region) < batch_size_pre_rank:
                        pop_num  = len(self._all_region)
                        [self._all_region.pop(0) for _ in range(pop_num)]
                    else:
                        self.order_episodes.extend([self._all_region.pop(0) for _ in range(batch_size_pre_rank)])
                        self.order_levels.extend(['region'] * batch_size_pre_rank)
                if self.objaverse_data.obj_ids:
                    if len(self.objaverse_data.obj_ids) < batch_size_pre_rank:
                        pop_num  = len(self.objaverse_data.obj_ids)
                        [self.objaverse_data.obj_ids.pop(0) for _ in range(pop_num)]
                    else:
                        self.order_episodes.extend([self.objaverse_data.obj_ids.pop(0) for _ in range(batch_size_pre_rank)])
                        self.order_levels.extend(['instance'] * batch_size_pre_rank)
            print_log(f'[DATASET] {len(self.order_episodes)} total samples were loaded', logger = 'SceneVerse')
            
    def _load_objaverse_data(self):
        self.objaverse_data = Objaverse(self._instance_npoint, 'train')
        print_log(f'[DATASET] {len(self.objaverse_data.obj_ids)} objects were loaded', logger = 'SceneVerse')
    
    def _load_aug_region_data(self):
        augment_region_data_dir = 'data/SceneVerse/RegionAugData'
        self._all_region = os.listdir(augment_region_data_dir)
        print_log(f'[DATASET] {len(self._all_region)} scans from RegionData were loaded', logger = 'SceneVerse')
    
    def _load_scan(self, pcd_path, inst2label_path, scan_name):
        pcd_data = torch.load(os.path.join(pcd_path, f'{scan_name}'))
        try:
            inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}"))
        except:
            inst_to_label = None
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    
        pcds = np.concatenate([points, colors], 1)
        return points, colors, pcds, instance_labels, inst_to_label
    
    def _load_scan_data(self, scan_name, dataset_name):
        dataset_root = os.path.join(self._all_dataset_root, dataset_name)
        annotation_root = os.path.join(dataset_root, 'annotations')
        scan_data_root = os.path.join(dataset_root, 'scan_data')
        
        inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
        pcd_path = os.path.join(scan_data_root,'pcd_with_global_alignment')

        points, colors, pcds, instance_labels, inst_to_label = self._load_scan(pcd_path, inst2label_path, scan_name)
        
        return points, colors, pcds, instance_labels, inst_to_label
        
    def convert_pc_to_box(self, obj_pc):
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
        box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
        return center, box_size
    
    def down_sample(self, points, colors, instance_labels=None, npoint=None):
        pcd_idxs = np.random.choice(len(points), size=npoint, replace=len(points) < npoint)
        points = points[pcd_idxs]
        colors = colors[pcd_idxs]
        instance_labels = instance_labels[pcd_idxs] if not instance_labels is None else None
        return points, colors, instance_labels
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def __len__(self):
        return len(self.order_episodes)
        
    def __getitem__(self, index):
        
        level = self.order_levels[index]
        data = self.order_episodes[index]
        if level == 'scene':
            dataset_name, scan_name = data
            points, colors, _, instance_labels, inst_to_label = self._load_scan_data(scan_name, dataset_name)
            points, colors, instance_labels = self.down_sample(points, colors, instance_labels, self._npoint)
            points = self.pc_norm(points)
            return f'{scan_name}@{level}', dataset_name, (points.astype(np.float32), self._num_groups, self._group_size)
        elif level == 'region':
            region_data = np.load(os.path.join('data/SceneVerse/RegionAugData', data))
            dataset_name = data.split('/')[-1].split('_')[0]
            scan_name = data.split('/')[-1].split('_')[1]
            points, colors, instance_labels = region_data['points'], region_data['colors'], region_data['instance_labels']
            points, colors, instance_labels = self.down_sample(points, colors, instance_labels, self._region_npoint)
            points = self.pc_norm(points)
            return f'{scan_name}@{level}', dataset_name, (points.astype(np.float32), self._region_num_groups, self._region_group_size)
        elif level == 'instance':
            obj_pcd = self.objaverse_data.load_obj_pcd(data)
            points = obj_pcd[:, :3]
            colors = obj_pcd[:, 3:]
            points, colors, _ = self.down_sample(points, colors, npoint=self._instance_npoint)
            points = self.pc_norm(points)
            return f'{data}@object', 'Objaverse', (points.astype(np.float32), self._instance_num_groups, self._instance_group_size)
            
            


    # def _get_inflate_axis_aligned_bounding_box(pcs, remaining_pcd, scale=scale, ):
    #     '''
    #         pcs & remaining_pcd : [N*3]
    #     '''
    #     import open3d
    #     tmp_pc = open3d.geometry.PointCloud()
    #     tmp_pc.points = open3d.utility.Vector3dVector(pcs)
    #     bbox = tmp_pc.get_axis_aligned_bounding_box()
    #     # 扩大边界框的尺寸n倍
    #     center = bbox.get_center()
    #     bbox.scale(scale, center)
    #     # 获取扩大后的边界框的最小和最大坐标
    #     min_bound = bbox.get_min_bound()
    #     max_bound = bbox.get_max_bound()
    #     # TODO: 这里的高度边界应该选择整个场景的边界
    #     min_bound[-1] = raw_point_cloud_dims_min[-1]
    #     max_bound[-1] = raw_point_cloud_dims_max[-1]
    #     # 选择边界框内的余下点云的点
    #     indices_within_bbox = []
    #     for i, point in enumerate(remaining_pcd):
    #         if np.all(min_bound <= point) and np.all(point <= max_bound):
    #             indices_within_bbox.append(i)
    #     return indices_within_bbox
    
    
@DATASETS.register_module()
class RegionVerseDataset(SceneVerseDataset):
    def __init__(self, config):
        super().__init__(config)
        self._npoint = config.N_POINTS
        self._num_groups = config.NUM_GROUP
        self._group_size = config.GROUP_SIZE
        if config.PREPROCESS:
            rank = (torch.distributed.get_rank())
            world_size = (torch.distributed.get_world_size())
            self.all_info = [(scan, dn) for scan, dn in zip(self._all_scans, self.dataset_names)]
            random.shuffle(self.all_info)
            
            self._all_scans = []
            self.dataset_names = []
            for info in self.all_info:
                self._all_scans.append(info[0])
                self.dataset_names.append(info[1])
                
            dist.broadcast_object_list(self._all_scans, src=0)
            dist.broadcast_object_list(self.dataset_names, src=0)
            
            if world_size > 1:
                total_scan_num = len(self._all_scans)
                scan_num_per_rank = int(total_scan_num // world_size)
                # Alloc the scans to each rank
                start_idx = rank * scan_num_per_rank
                end_idx = (rank + 1) * scan_num_per_rank if rank != world_size - 1 else total_scan_num
                print(f'Rank {rank} has {end_idx - start_idx} scans from {start_idx} to {end_idx}')
                self._all_scans = self._all_scans[start_idx:end_idx]
                self.dataset_names = self.dataset_names[start_idx:end_idx]
            self._preprocess_scene_to_region()
            torch.distributed.barrier()
            assert False
        else:
            self._load_region_from_disk()   
        print_log(f'[DATASET] {len(self._all_region)} regions were loaded', logger = 'RegionVerse')
        
    def _preprocess_scene_to_region(self):
        save_dir = 'data/SceneVerse/RegionAugData'
        os.makedirs(save_dir, exist_ok=True)
        self._all_region = []
        SAMPLE_REGION_PER_SCAN = 10
        pbar = tqdm(total=len(self._all_scans))
        for dataset_name, scan_name in (zip(self.dataset_names, self._all_scans)):
            # print(dataset_name)
            # print(scan_name)
            pbar.update(1)
            points, colors, pcds, instance_labels, inst_to_label = self._load_scan_data(scan_name, dataset_name)
            
            if instance_labels is None:
                continue
        
            # if len(points) < int(SAMPLE_NPOINT_PRE_REGION * 2):
            #     continue
            # kd_tree = cKDTree(points)
            
            # Visualization the scene level group
            """ from utils import misc
            ds_point = misc.fps(torch.from_numpy(points).unsqueeze(0).float().cuda(), self._npoint).cpu().numpy()[0]
            kd_tree = cKDTree(ds_point)
            
            dvae_colors = np.ones_like(ds_point) * 0.5
            center = misc.fps(torch.from_numpy(ds_point).unsqueeze(0).float().cuda(), self._num_groups).cpu().numpy()
            for c in center[0]:
                distances, indices = kd_tree.query(c, k=self._group_size)
                dvae_colors[indices] = np.repeat(np.random.rand(1, 3), repeats=self._group_size, axis=0)
            visualization_pointclouds(ds_point, dvae_colors)
            continue """
        
            # visualization_pointclouds(points, colors / 255)
            inst_ids = np.unique(instance_labels).astype(np.uint64)
            valid_instance_ids = []
            for inst_id in inst_ids:
                
                inst_id = int(inst_id)
                
                if inst_id == 0:
                    continue
                
                inst_pc = points[instance_labels == inst_id]
                
                point_threshold = 500
                if not inst_pc.shape[0] > point_threshold:
                    continue
                
                valid_instance_ids.append(inst_id)

                save_path = os.path.join(save_dir, f'{dataset_name}_{scan_name}_{inst_id}.npz')
                if os.path.exists(save_path):
                    continue
                
                
                ''' 1. Random sample instance points 2. Sample nearest N point use KDTree
                inst_pc_id = np.random.choice(inst_pc.shape[0])
                distances, indices = kd_tree.query(inst_pc[inst_pc_id], k=SAMPLE_NPOINT_PRE_REGION)
                region_points = points[indices]
                region_colors = colors[indices]
                region_instance_labels = instance_labels[indices]
                '''
                
                # ''' 1. Random sample one instance 2. Comput instance bounding box 3. Sample points in the scale bounding box
                # import open3d
                # tmp_pc = open3d.geometry.PointCloud()
                # tmp_pc.points = open3d.utility.Vector3dVector(inst_pc)

                # bbox = tmp_pc.get_axis_aligned_bounding_box()
                # whl = bbox.get_max_bound() - bbox.get_min_bound()
                # bbox_size = whl[0] * whl[1] * whl[2]
                # As we want the sample region's size between 1 m3 and 2 m3
                # if bbox_size > 2:
                #     continue
                # print(bbox_size)
                # if bbox_size <= 0.001:
                #     scale = (1024 * (0.001 - bbox_size) * 10000)**(1/3)
                # elif 0.01 >= bbox_size > 0.001:
                #     scale = (128 * (0.01 - bbox_size) * 1000)**(1/3)
                # elif 0.1 >= bbox_size > 0.01:
                #     scale = (16 * (0.1 - bbox_size) * 100)**(1/3)
                # elif 1 >=  bbox_size > 0.1:
                #     scale = (2 * (1 - bbox_size) * 10)**(1/3)
                # elif bbox_size > 1:
                #     scale = 1
                # # print(scale)
                # # Scale bbox to N bbox
                # bbox.scale(scale)
                # min_bound = bbox.get_min_bound()
                # max_bound = bbox.get_max_bound()
                # whl = bbox.get_max_bound() - bbox.get_min_bound()
                # scale_bbox_size = whl[0] * whl[1] * whl[2]
                # print(scale_bbox_size)
                
                # raw_point_cloud_dims_min = points[..., :3].min(axis=0)
                # raw_point_cloud_dims_max = points[..., :3].max(axis=0)
                # min_bound[-1] = raw_point_cloud_dims_min[-1]
                # max_bound[-1] = raw_point_cloud_dims_max[-1]
                
                fix_whl_size = 2
                center, whl = self.convert_pc_to_box(inst_pc)
                fixed_whl = np.array([fix_whl_size, fix_whl_size, fix_whl_size])  

                min_bound = center - fixed_whl / 2.0
                max_bound = center + fixed_whl / 2.0
                
                indices_within_bbox = []
                for i, point in enumerate(points):
                    if np.all(min_bound <= point) and np.all(point <= max_bound):
                        indices_within_bbox.append(i)
                region_points = points[indices_within_bbox]
                region_colors = colors[indices_within_bbox]
                region_instance_labels = instance_labels[indices_within_bbox]
                # '''
                
                # print(len(region_points))
                # print(inst_to_label[inst_id])
                
                if not region_points.shape[0] > point_threshold:
                    print("err")
                    continue
                
                # visualization_pointclouds(region_points, region_colors / 255)
                
                # Save region data here
                np.savez_compressed(save_path, 
                    points=region_points, colors=region_colors, instance_labels=region_instance_labels)
                
                self._all_region.append((region_points, region_colors, region_instance_labels))
    
    def __len__(self):
        return len(self._all_region)
    
    def __getitem__(self, index):
        region_points, region_colors, region_instance_labels = self._all_region[index]
        
        points, colors, instance_labels = self.down_sample(points, colors, instance_labels)
        region_points = self.pc_norm(region_points)
        return None, None, (region_points, self._num_groups, self._group_size)
    
    def down_sample(self, points, colors, instance_labels):
        pcd_idxs = np.random.choice(len(points), size=self._npoint, replace=len(points) < self._npoint)
        points = points[pcd_idxs]
        colors = colors[pcd_idxs]
        instance_labels = instance_labels[pcd_idxs]
        return points, colors, instance_labels


def visualization_pointclouds(pc, color):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])
    
    
@DATASETS.register_module()
class SceneVerseLLMPretrainDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf', add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        
        # Load scene level data
        self._npoint = config.N_POINTS
        self._group_size = config.GROUP_SIZE
        self._num_groups = config.NUM_GROUP
        _all_dataset_name = ['3RScan', 'HM3D', 'MultiScan', 'ARKitScenes', 'ScanNet']
    
        self._all_dataset_root = 'data/SceneVerse'
        self._all_scans = []
        self.dataset_names = []
        
        # Step1: Load all type of annotations from all scans in ${all_dataset_dict}
        all_dataset_dict = {}
        for dataset_name in _all_dataset_name:
            path = f'{self._all_dataset_root}/{dataset_name}/scan_data/pcd_with_global_alignment'
            
            # Dict: {object_caption: {}, scene_caption: {}, relation_caption: {}}
            dataset_anno_dict = self._load_annotation(f'{self._all_dataset_root}/{dataset_name}/annotations')
            all_dataset_dict[dataset_name] = dataset_anno_dict
            
            # Filter no pcd data scan
            if not os.path.exists(path):
                continue
            
            # Load all scan pcd files
            data = glob.glob(f'{path}/*.pth')
            data = [d.split('/')[-1] for d in data]
            self.dataset_names.extend([dataset_name] * len(data))
            self._all_scans.extend(data)
            print_log(f'[DATASET] {len(data)} scans from {dataset_name} were loaded', logger = 'SceneVerse')
        self._all_scans_datasets = []
        for dn, sc in zip(self.dataset_names, self._all_scans):
            self._all_scans_datasets.append((dn, sc))
        
        # Step2: Link all annotations to the corresponding scan pcd
        self.all_scene_caption = []
        self.all_object_caption = []
        self.all_relation_caption = []
        
        region_aug_data_dir = 'data/SceneVerse/RegionAugData'
        region_aug_pcd = os.listdir(region_aug_data_dir)
        # As dict search key will fast
        region_aug_pcd = {p:[] for p in region_aug_pcd}
        for dataset_name, annotations in all_dataset_dict.items():
            # Process scene caption: {scan_name: [captions]}
            # ~10k
            scene_annos = annotations.get('scene_caption')
            for scan_name, captions in scene_annos.items():
                self.all_scene_caption.extend([{'dataset_name':dataset_name, "scan_name":scan_name, "anno":{'utterance':cap}, "task_name": "scene_caption"} for cap in captions['captions']])
            
            # Process region caption: [dict_keys(['item_id', 'scan_id', 'target_id', 'instance_type', 'utterance']), xxx]
            # ~370k
            region_annos = annotations.get('relation_caption')
            ## Check if the pcd corresponding to the region caption exists
            for ra in region_annos:
                scan_name = ra['scan_id']
                if f'{dataset_name}_{scan_name}.pth_{ra["target_id"]}.npz' not in region_aug_pcd.keys():
                    continue
                self.all_relation_caption.append({'dataset_name':dataset_name, "scan_name":scan_name, "anno":ra, "task_name": "region_caption"})
            
            # Process object caption: [dict_keys(['item_id', 'scan_id', 'target_id', 'instance_type', 'utterance']), xxx]
            # ~112k
            object_annos = annotations.get('object_caption')
            for oa in object_annos:
                scan_name = oa['scan_id']
                # As some object instance pointcloud will miss after downsample, we only keep the object caption with pcd data large than threshold
                if f'{dataset_name}_{scan_name}.pth_{oa["target_id"]}.npz' not in region_aug_pcd.keys():
                    continue
                self.all_object_caption.append({'dataset_name':dataset_name, "scan_name":scan_name, "anno":oa, "task_name": "object_caption"})

        print_log(f'[DATASET] {len(self.all_scene_caption)} scene captions were loaded from scan data', logger = 'SceneVerse')
        print_log(f'[DATASET] {len(self.all_relation_caption)} relation captions were loaded from scan data', logger = 'SceneVerse')
        print_log(f'[DATASET] {len(self.all_object_caption)} object captions were loaded from scan data', logger = 'SceneVerse')
        
        # Load region level data
        self._region_npoint = config.REGION_N_POINTS
        self._region_group_size = config.REGION_GROUP_SIZE
        self._region_num_groups = config.REGION_NUM_GROUP
        
        # Augment object caption task into object grouding task
        all_grouding_object = copy.deepcopy(self.all_object_caption)
        for data in all_grouding_object:
            data['task_name'] = 'object_grouding'
        self.all_scene_caption.extend(all_grouding_object)
        
        # Load some objaverse caption data 
        self._instance_npoint = config.INSTANCE_N_POINTS
        self._instance_group_size = config.INSTANCE_GROUP_SIZE
        self._instance_num_groups = config.INSTANCE_NUM_GROUP
        self._load_objaverse_data()
        
        for obj_dict in list(self.objaverse_data.obj_cap_dict.keys())[:200000]:
            self.all_object_caption.append({'dataset_name':'Objaverse', 
                                            "scan_name":obj_dict, 
                                            "task_name": "object_caption",
                                            "anno":{
                                                    'item_id': obj_dict,
                                                    'scan_id': obj_dict,
                                                    'utterance': self.objaverse_data.obj_cap_dict[obj_dict]
                                                    }})
        print_log(f'[DATASET] {len(self.all_object_caption)} object captions were loaded from scan data and objaverse', logger = 'SceneVerse')
        
        random.seed(0)
        random.shuffle(self.all_scene_caption)
        # Drop to 200k
        random.shuffle(self.all_relation_caption)
        self.all_relation_caption = self.all_relation_caption[:200000]
        random.shuffle(self.all_object_caption)
        dist.broadcast_object_list(self.all_scene_caption, src=0)
        dist.broadcast_object_list(self.all_relation_caption, src=0)
        dist.broadcast_object_list(self.all_object_caption, src=0)
        
        # if config.subset == 'train':
        #     self.all_scene_caption = self.all_scene_caption[:-1000]
        #     self.all_relation_caption = self.all_relation_caption[:-10000]
        #     self.all_object_caption = self.all_object_caption[:-10000]
        # else:
        #     self.all_scene_caption = self.all_scene_caption[-1000:]
        #     self.all_relation_caption = self.all_relation_caption[-10000:]
        #     self.all_object_caption = self.all_object_caption[-10000:]           

        # As diffent dataset has different number of points, we need to specify the dataset squence order 
        # to make sure samples from on batch come from the same level dataset
        batch_size_pre_rank = config.tbs
        self.order_episodes = []
        self.order_levels = []
        
        # Debug
        self.all_scene_caption = []
        self.all_object_caption = []
        
        while self.all_scene_caption or self.all_relation_caption or self.all_object_caption:
            if self.all_scene_caption:
                if len(self.all_scene_caption) < batch_size_pre_rank:
                    pop_num  = len(self.all_scene_caption)
                    [self.all_scene_caption.pop(0) for _ in range(pop_num)]
                else:
                    self.order_episodes.extend([self.all_scene_caption.pop(0) for _ in range(batch_size_pre_rank)])
                    self.order_levels.extend(['scene'] * batch_size_pre_rank)
            if self.all_relation_caption:
                if len(self.all_relation_caption) < batch_size_pre_rank:
                    pop_num  = len(self.all_relation_caption)
                    [self.all_relation_caption.pop(0) for _ in range(pop_num)]
                else:
                    self.order_episodes.extend([self.all_relation_caption.pop(0) for _ in range(batch_size_pre_rank)])
                    self.order_levels.extend(['region'] * batch_size_pre_rank)
            if self.all_object_caption:
                if len(self.all_object_caption) < batch_size_pre_rank:
                    pop_num  = len(self.all_object_caption)
                    [self.all_object_caption.pop(0) for _ in range(pop_num)]
                else:
                    self.order_episodes.extend([self.all_object_caption.pop(0) for _ in range(batch_size_pre_rank)])
                    self.order_levels.extend(['instance'] * batch_size_pre_rank)
        print_log(f'[DATASET] {len(self.order_episodes)} total samples were loaded for split {config.subset}', logger = 'SceneVerse')
       
    
    def _load_annotation(self, annotation_path):
        dataset_name_to_annotation = {
            'object_caption' : ('ssg_obj_caption_gpt.json', 'ssg_obj_caption_template.json'),
            'scene_caption' : ('scene_cap.json', ),
            'relation_caption': ('ssg_ref_chain_gpt.json', 'ssg_ref_relm_gpt.json')
        }
        output_annotation = {}
        for k, v in dataset_name_to_annotation.items():
            for fn in v:
                if annotation_path.find('ScanNet') != -1:
                    fp = os.path.join(annotation_path, 'refer', fn)
                else:
                    fp = os.path.join(annotation_path, fn)
                if not os.path.exists(fp):
                    continue
                with open(fp, 'r') as f:
                    data = json.load(f)
                    if annotation_path.find('ScanNet') != -1 and k == 'scene_caption':
                        data = {d['scene_id']:{"captions":d['answers']} for d in data}
                    output_annotation[k] = data
                break
        return output_annotation

    def _load_objaverse_data(self):
        self.objaverse_data = Objaverse(self._instance_npoint, 'train')
        print_log(f'[DATASET] {len(self.objaverse_data.obj_ids)} objects were loaded', logger = 'SceneVerse')
    
    def _load_scan(self, pcd_path, inst2label_path, scan_name):
        pcd_data = torch.load(os.path.join(pcd_path, f'{scan_name}'))
        try:
            inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}"))
        except:
            inst_to_label = None
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    
        pcds = np.concatenate([points, colors], 1)
        return points, colors, pcds, instance_labels, inst_to_label
    
    def _load_scan_data(self, scan_name, dataset_name):
        dataset_root = os.path.join(self._all_dataset_root, dataset_name)
        annotation_root = os.path.join(dataset_root, 'annotations')
        scan_data_root = os.path.join(dataset_root, 'scan_data')
        
        inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
        pcd_path = os.path.join(scan_data_root,'pcd_with_global_alignment')

        points, colors, pcds, instance_labels, inst_to_label = self._load_scan(pcd_path, inst2label_path, scan_name)
        
        return points, colors, pcds, instance_labels, inst_to_label
        
    def convert_pc_to_box(self, obj_pc):
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
        box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
        return center, box_size
    
    def down_sample(self, points, colors, instance_labels=None, npoint=None):
        pcd_idxs = np.random.choice(len(points), size=npoint, replace=len(points) < npoint)
        points = points[pcd_idxs]
        colors = colors[pcd_idxs]
        instance_labels = instance_labels[pcd_idxs] if not instance_labels is None else None
        return points, colors, instance_labels
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def __len__(self):
        return len(self.order_episodes)
    
    def _encode_box_coords(self, gt_box_centers_normalized, gt_box_sizes_normalized):
        grid_size_3d = 255
        BOX_FORMAT = '<obj>{}, {}, {}, {}, {}, {}</obj>'
        center_normalized = gt_box_centers_normalized
        size_normalized = gt_box_sizes_normalized
        box_normalized = np.hstack((center_normalized, size_normalized))    # (-1, 6)
        # <cx, cy, cz, w, h, l>
        box_normalized = (box_normalized * grid_size_3d).astype(np.int64)
        return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)
    
    def _scale_points(self, pred_xyz, mult_factor):
        if pred_xyz.ndim == 4:
            mult_factor = mult_factor[:, None]
        scaled_xyz = pred_xyz * mult_factor[:, None, :]
        return scaled_xyz
    
    def _shift_scale_points(self, pred_xyz, src_range, dst_range=None):
        """
        pred_xyz: B x N x 3
        src_range: [[B x 3], [B x 3]] - min and max XYZ coords
        dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
        """
        if dst_range is None:
            dst_range = [
                torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
                torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
            ]

        if pred_xyz.ndim == 4:
            src_range = [x[:, None] for x in src_range]
            dst_range = [x[:, None] for x in dst_range]

        assert src_range[0].shape[0] == pred_xyz.shape[0]
        assert dst_range[0].shape[0] == pred_xyz.shape[0]
        assert src_range[0].shape[-1] == pred_xyz.shape[-1]
        assert src_range[0].shape == src_range[1].shape
        assert dst_range[0].shape == dst_range[1].shape
        assert src_range[0].shape == dst_range[1].shape

        src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
        dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
        prop_xyz = (
            ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
        ) + dst_range[0][:, None, :]
        return prop_xyz
    
    def __getitem__(self, index):
        
        level = self.order_levels[index]
        data = self.order_episodes[index]
        dataset_name, scan_name, anno, task_name = data['dataset_name'], data['scan_name'], data['anno'], data['task_name']
        
        self.tokenizer_config = dict(
            max_length=128, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        
        if level == 'scene':
            points, colors, _, instance_labels, inst_to_label = self._load_scan_data(f'{scan_name}.pth', dataset_name)
            points, colors, instance_labels = self.down_sample(points, colors, instance_labels, self._npoint)
            points = self.pc_norm(points)
            
            if task_name == 'object_grouding':
                tgt_id = int(anno['target_id'])
                object_points = points[instance_labels == tgt_id]
                center, whl = self.convert_pc_to_box(object_points)
                
                point_cloud_dims_min = points.min(axis=0)
                point_cloud_dims_max = points.max(axis=0)

                box_centers = center.astype(np.float32)
                
                center_normalizing_range = [
                    np.zeros((1, 3), dtype=np.float32),
                    np.ones((1, 3), dtype=np.float32),
                ]
                box_centers_normalized = self._shift_scale_points(
                    box_centers[None, ...],
                    src_range=[
                        point_cloud_dims_min[None, ...],
                        point_cloud_dims_max[None, ...],
                    ],
                    dst_range=center_normalizing_range,
                )
                mult_factor = point_cloud_dims_max - point_cloud_dims_min
                box_sizes_normalized = self._scale_points(
                    whl.astype(np.float32)[None, ...],
                    mult_factor=1.0 / mult_factor[None, ...],
                )
                bbox_str = self._encode_box_coords(box_centers_normalized[0], box_sizes_normalized[0])
                
                ret_dict = {
                    'points': points.astype(np.float32),
                    'colors': colors.astype(np.float32),
                    'num_groups': self._num_groups,
                    'group_size': self._group_size,
                    'dataset_name': dataset_name,
                    'level': level,
                    'scan_name': scan_name,
                }
                
                caption = anno['utterance']
                intruction = f'Locate the object in the scene according to the object caption: {caption}'
                prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
                
                answers = bbox_str
                llm_inputs = self.tokenizer.batch_encode_plus(
                [' '.join((intruction, answers, self.tokenizer.eos_token))],
                **self.tokenizer_config
                )
                
                ret_dict['answers'] = answers
                ret_dict['question'] = intruction
                ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
                ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
                ret_dict['gradient_mask'] = \
                    (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
                ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
                ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)

                return ret_dict
            
            # Scene Caption
            else:
                ret_dict = {
                    'points': points.astype(np.float32),
                    'colors': colors.astype(np.float32),
                    'num_groups': self._num_groups,
                    'group_size': self._group_size,
                    'dataset_name': dataset_name,
                    'level': level,
                    'scan_name': scan_name,
                }
                
                intruction = f'Describe the scene in detailed: '
                prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
                
                answers = anno['utterance']
                llm_inputs = self.tokenizer.batch_encode_plus(
                [' '.join((intruction, answers, self.tokenizer.eos_token))],
                **self.tokenizer_config
                )
                
                ret_dict['answers'] = answers
                ret_dict['question'] = intruction
                ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
                ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
                ret_dict['gradient_mask'] = \
                    (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
                ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
                ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)

                return ret_dict
            
        
        elif level == 'region':
            instance_id = int(anno['target_id'])
            path = f'{dataset_name}_{scan_name}.pth_{instance_id}.npz'
            region_data = np.load(os.path.join('data/SceneVerse/RegionAugData', path))
            points, colors, instance_labels = region_data['points'], region_data['colors'], region_data['instance_labels']
            # points, colors, instance_labels = self.down_sample(points, colors, instance_labels, self._region_npoint)
            # points = self.pc_norm(points)
            
            ret_dict = {
                'points': points.astype(np.float32),
                'colors': colors.astype(np.float32),
                'num_groups': self._region_num_groups,
                'group_size': self._region_group_size,
                'dataset_name': dataset_name,
                'level': level,
                'scan_name': scan_name,
            }
            
            intruction = 'Describe the relation of these objects: '
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            
            answers = anno['utterance']
            llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
            )
            
            ret_dict['answers'] = answers
            ret_dict['question'] = intruction
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)

            
            print(answers)            
            visualization_pointclouds(points, colors / 255)
            
            return ret_dict
        
        elif level == 'instance':
            if dataset_name == 'Objaverse':
                obj_pcd = self.objaverse_data.load_obj_pcd(scan_name)
                points = obj_pcd[:, :3]
                colors = obj_pcd[:, 3:]
                points, colors, _ = self.down_sample(points, colors, npoint=self._instance_npoint)
                points = self.pc_norm(points)
            else:
                points, colors, _, instance_labels, inst_to_label = self._load_scan_data(f'{scan_name}.pth', dataset_name)
                instance_id = int(anno['target_id'])
                object_points = points[instance_labels == instance_id]
                object_colors = colors[instance_labels == instance_id]
                instance_labels = instance_labels[instance_labels == instance_id]
                points, colors, _ = self.down_sample(object_points, object_colors, instance_labels, npoint=self._instance_npoint)
                points = self.pc_norm(points)
            
            ret_dict = {
                'points': points.astype(np.float32),
                'colors': colors.astype(np.float32),
                'num_groups': self._instance_num_groups,
                'group_size': self._instance_group_size,
                'dataset_name': dataset_name,
                'level': level,
                'scan_name': scan_name,
            }
            
            intruction = 'Describe the object: '
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            
            answers = anno['utterance']
            llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
            )
            
            ret_dict['answers'] = answers
            ret_dict['question'] = intruction
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)

            return ret_dict
            