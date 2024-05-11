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
                self._all_scans_datasets = self._all_scans_datasets[:-10000]
                self._all_region = self._all_region[:-10000]
                self.objaverse_data.obj_ids = self.objaverse_data.obj_ids[:200000]
            else:
                self._all_scans_datasets = self._all_scans_datasets[-10000:]
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
        augment_region_data_dir = 'data/SceneVerse/RegionData'
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
            return f'{scan_name}_{level}', dataset_name, (points.astype(np.float32), self._num_groups, self._group_size)
        elif level == 'region':
            region_data = np.load(os.path.join('data/SceneVerse/RegionData', data))
            dataset_name = data.split('/')[-1].split('_')[0]
            scan_name = data.split('/')[-1].split('_')[1]
            points, colors, instance_labels = region_data['points'], region_data['colors'], region_data['instance_labels']
            points, colors, instance_labels = self.down_sample(points, colors, instance_labels, self._region_npoint)
            points = self.pc_norm(points)
            return f'{scan_name}_{level}', dataset_name, (points.astype(np.float32), self._region_num_groups, self._region_group_size)
        elif level == 'instance':
            obj_pcd = self.objaverse_data.load_obj_pcd(data)
            points = obj_pcd[:, :3]
            colors = obj_pcd[:, 3:]
            points, colors, _ = self.down_sample(points, colors, npoint=self._instance_npoint)
            points = self.pc_norm(points)
            return f'{data}_object', 'Objaverse', (points.astype(np.float32), self._instance_num_groups, self._instance_group_size)
            
            


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
        save_dir = 'data/SceneVerse/RegionData'
        os.makedirs(save_dir, exist_ok=True)
        self._all_region = []
        SAMPLE_REGION_PER_SCAN = 10
        SAMPLE_REGION_RATIO = 0.2
        SAMPLE_NPOINT_PRE_REGION = 10000
        pbar = tqdm(total=len(self._all_scans))
        for dataset_name, scan_name in (zip(self.dataset_names, self._all_scans)):
            # print(dataset_name)
            # print(scan_name)
            pbar.update(1)
            points, colors, pcds, instance_labels, inst_to_label = self._load_scan_data(scan_name, dataset_name)
        
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
            
            inst_ids = np.unique(instance_labels)
            for ri in range(SAMPLE_REGION_PER_SCAN):
                inst_id = np.random.choice(inst_ids)
                inst_pc = points[instance_labels == inst_id]
                
                if not len(inst_pc) > 500:
                    continue
                
                save_path = os.path.join(save_dir, f'{dataset_name}_{scan_name}_{ri}.npz')
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
                import open3d
                tmp_pc = open3d.geometry.PointCloud()
                tmp_pc.points = open3d.utility.Vector3dVector(inst_pc)

                bbox = tmp_pc.get_axis_aligned_bounding_box()
                whl = bbox.get_max_bound() - bbox.get_min_bound()
                bbox_size = whl[0] * whl[1] * whl[2]
                # As we want the sample region's size between 1 m3 and 2 m3
                if bbox_size > 2:
                    continue
                # print(bbox_size)
                if bbox_size <= 0.001:
                    scale = (1024 * (0.001 - bbox_size) * 10000)**(1/3)
                elif 0.01 >= bbox_size > 0.001:
                    scale = (128 * (0.01 - bbox_size) * 1000)**(1/3)
                elif 0.1 >= bbox_size > 0.01:
                    scale = (16 * (0.1 - bbox_size) * 100)**(1/3)
                elif 1 >=  bbox_size > 0.1:
                    scale = (2 * (1 - bbox_size) * 10)**(1/3)
                elif bbox_size > 1:
                    scale = 1
                # print(scale)
                # Scale bbox to N bbox
                bbox.scale(scale)
                min_bound = bbox.get_min_bound()
                max_bound = bbox.get_max_bound()
                whl = bbox.get_max_bound() - bbox.get_min_bound()
                scale_bbox_size = whl[0] * whl[1] * whl[2]
                # print(scale_bbox_size)
                
                # raw_point_cloud_dims_min = points[..., :3].min(axis=0)
                # raw_point_cloud_dims_max = points[..., :3].max(axis=0)
                # min_bound[-1] = raw_point_cloud_dims_min[-1]
                # max_bound[-1] = raw_point_cloud_dims_max[-1]
                indices_within_bbox = []
                for i, point in enumerate(points):
                    if np.all(min_bound <= point) and np.all(point <= max_bound):
                        indices_within_bbox.append(i)
                region_points = points[indices_within_bbox]
                region_colors = colors[indices_within_bbox]
                region_instance_labels = instance_labels[indices_within_bbox]
                # '''
                
                # print(len(region_points))
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