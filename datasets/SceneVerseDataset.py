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
from copy import deepcopy
from scipy.spatial import cKDTree
from tqdm import tqdm
import random
import copy
import json
from transformers import AutoTokenizer

SYSTEM_PROMPT = "A chat between a curious human and an artificial intelligence assistant.\
The assistant gives helpful, detailed, and polite answers to the human's questions.\
The visual content will be provided with the following format: <scene>visual content</scene> and\
the object bounding box will be provided with the following format: <obj>x,y,z,width,height,length</obj>\n"

TASK_PROMPT = {
    # 'object_caption': [
    #     '### human: describe this object ### assistant:',
    #     # '### human: provide a brief description of the given 3D object. ### assistant:',
    #     # '### human: offer a succinct explanation of the 3D object presented. ### assistant:',
    #     # '### human: give a short and clear explanation of the 3D object. ### assistant:',
    #     # '### human: describe the characteristics of this object ### assistant:',
    #     # '### human: give an overview of the object. ### assistant:',
    #     # '### human: explain the features of the object ### assistant:',
    #     # '### human: provide a concise description of the object. ### assistant:',
    #     # '### human: detail the attributes of this object. ### assistant:',
    # ],
    # 'scene_caption': [
    #     '### human: describe the scene. ### assistant:',
    #     # '### human: provide a description of the scene in the given 3D context. ### assistant:',
    #     # '### human: explain the details of the scene within the specified 3D context. ### assistant:',
    #     # '### human: offer an overview of the scene in the indicated 3D context. ### assistant:',
    #     # '### human: describe the features of the scene in the provided 3D context. ### assistant:',
    #     # '### human: detail the elements of the scene in the mentioned 3D context. ### assistant:',
    #     # '### human: give an explanation of the scene in the designated 3D context. ### assistant:',
    #     # '### human: summarize the scene in the described 3D context. ### assistant:',
    #     # '### human: illustrate the scene within the given 3D context. ### assistant:',
    #     # '### human: characterize the scene in the specific 3D context. ### assistant:',
    # ],
    # 'region_caption': [
    #     # '### human: describe the position of the {object_name} in relation to the surrounding objects. ### assistant:',
    #     '### human: describe how the objects are related in the specified part of the 3D scene. ### assistant:',
    #     # '### human: provide details on the relationship between the objects in the indicated part of the 3D scene. ### assistant:',
    #     # '### human: offer an explanation of the relationship between the objects in the described part of the 3D scene. ### assistant:',
    #     # '### human: detail the interaction between the objects in the given segment of the 3D scene. ### assistant:',
    #     # '### human: illustrate the relationship between the objects within the specified part of the 3D scene. ### assistant:',
    #     # '### human: summarize the connection between the objects in the mentioned part of the 3D scene. ### assistant:',
    #     # '### human: characterize the relationship between the objects in the provided part of the 3D scene. ### assistant:',
    #     # '### human: outline the relationship between the objects in the designated part of the 3D scene. ### assistant:',
    #     # '### human: depict the interaction between the objects in the stated part of the 3D scene. ### assistant:',
    # ],
    # 'object_grouding': [
    #     '### human: locate the object in the 3D scene given the object description {caption}. ### assistant:',
    #     # '### human: find the object in the 3D scene using the provided description {caption}. ### assistant:',
    #     # '### human: identify the object in the 3D scene based on the given description {caption}. ### assistant:',
    #     # '### human: pinpoint the object in the 3D scene from the described details {caption}. ### assistant:',
    #     # '### human: determine the location of the object in the 3D scene using the description {caption}. ### assistant:',
    #     # '### human: locate the object within the 3D scene according to the description {caption}. ### assistant:',
    #     # '### human: ascertain the position of the object in the 3D scene given the description {caption}. ### assistant:',
    #     # '### human: find the position of the object in the 3D scene using the details provided {caption}. ### assistant:',
    #     # '### human: spot the object in the 3D scene based on the description {caption}. ### assistant:',
    #     # '### human: seek out the object in the 3D scene from the given details {caption}. ### assistant:',
    #     # '### human: identify where the object is located in the 3D scene using the description {caption}. ### assistant:',
    # ],
    # 'scene_grouding':[
    #     '### human: locate all objects given the 3D scene. ### assistant:',
    #     '### human: identify all objects in the given 3D scene. ### assistant:',
    #     '### human: find every object in the provided 3D scene. ### assistant:',
    #     '### human: list all objects present in the specified 3D scene. ### assistant:',
    #     '### human: pinpoint each object in the given 3D scene. ### assistant:',
    #     '### human: locate every object in the described 3D scene. ### assistant:',
    #     '### human: determine the location of all objects in the provided 3D scene. ### assistant:',
    #     '### human: identify the position of each object in the given 3D scene. ### assistant:',
    #     '### human: find and list all objects in the specified 3D scene. ### assistant:',
    #     '### human: recognize all objects within the given 3D scene. ### assistant:',
    #     '### human: enumerate every object in the provided 3D scene. ### assistant:',
    # ],
    # 'object_caption_given_bbox':[
    #     '### human: describe the object in the 3D scene given the object bounding box {bbox}. ### assistant:',
    #     # '### human: provide a description of the object in the 3D scene using the given bounding box {bbox}. ### assistant:',
    #     # '### human: describe the object within the 3D scene based on the provided bounding box {bbox}. ### assistant:',
    #     # '### human: explain the features of the object in the 3D scene given its bounding box {bbox}. ### assistant:',
    #     # '### human: detail the characteristics of the object in the 3D scene using the bounding box {bbox}. ### assistant:',
    #     # '### human: offer a description of the object in the 3D scene from the given bounding box {bbox}. ### assistant:',
    #     # '### human: summarize the attributes of the object in the 3D scene using the provided bounding box {bbox}. ### assistant:',
    #     # '### human: characterize the object in the 3D scene based on its bounding box {bbox}. ### assistant:',
    #     # '### human: give an overview of the object in the 3D scene using the bounding box information {bbox}. ### assistant:',
    #     # '### human: identify the features of the object in the 3D scene given the bounding box {bbox}. ### assistant:',
    #     # '### human: outline the details of the object in the 3D scene using the provided bounding box {bbox}. ### assistant:',
    # ],
    # 'scene_qa':[
    #     '### human: answer the question about the 3D scene in short. {question} ### assistant:',
    #     # '### human: give a brief answer to the question about the 3D scene. {question} ### assistant:',
    #     # '### human: provide a short response to the 3D scene question. {question} ### assistant:',
    #     # '### human: respond concisely to the inquiry regarding the 3D scene. {question} ### assistant:',
    #     # '### human: offer a succinct answer to the 3D scene question. {question} ### assistant:',
    #     # '### human: briefly answer the question about the 3D scene. {question} ### assistant:',
    #     # '### human: give a short response to the question concerning the 3D scene. {question} ### assistant:',
    #     # '### human: provide a concise reply to the question about the 3D scene. {question} ### assistant:',
    #     # '### human: answer the inquiry on the 3D scene briefly. {question} ### assistant:',
    #     # '### human: give a brief response to the question related to the 3D scene. {question} ### assistant:',
    #     # '### human: offer a short answer to the question regarding the 3D scene. {question} ### assistant:',
    # ],
    'object_caption': [
        dict(
            instruction='### human: given the 3D scene, describe this object. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: describe this object in the given 3D scene. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, localize and describe this object. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: localize and describe this object in the given 3D scene. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, describe this object first, then localize it. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
        dict(
            instruction='### human: describe then localize the object from the 3D scene. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
    ],
    'scene_qa': [
        dict(
            instruction='### human: given the 3D scene, answer the question: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer this quesiton according to the given 3D scene: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer the question: "{question}" with the related object locations in the input 3D scene. ### assistant:',
            answer='the answer is: {answer}, and the related objects are localized at {locations}',
            do_localize=True
        ),
        dict(
            instruction='### human: given the 3D scene, localize all the related objects first, then answer the question: "{question}" ### assistant:',
            answer='the related objects are localized at {locations}, the answer is: {answer}',
            do_localize=True
        ),
    ],
    'hd_scene_qa': [
        dict(
            instruction='### human: based on the 3D scene with multiple rooms, answer the question: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer this question based on the provided multi-room 3D scene: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer the question: "{question}" using the relevant object locations from the provided multi-room 3D scene. ### assistant:',
            answer='the answer is: {answer}, and the related objects are localized at {locations}',
            do_localize=True
        ),
        dict(
            instruction='### human: given the multi-room 3D scene, first locate all the relevant objects, then answer the question: "{question}" ### assistant:',
            answer='the related objects are localized at {locations}, the answer is: {answer}',
            do_localize=True
        ),
    ],
    'nuscenes_qa': [
        dict(
            instruction='### human: based on the outdoor scene, answer the question: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer this question based on the provided outdoor scene: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer the question: "{question}" using the relevant object locations from the provided outdoor scene. ### assistant:',
            answer='the answer is: {answer}, and the related objects are localized at {locations}',
            do_localize=True
        ),
        dict(
            instruction='### human: given the outdoor scene, first locate all the relevant objects, then answer the question: "{question}" ### assistant:',
            answer='the related objects are localized at {locations}, the answer is: {answer}',
            do_localize=True
        ),
    ],
    'nuscenes_object_caption': [
        dict(
            instruction='### human: given the outdoor scene, describe this object. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: describe this object in the given outdoor scene. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, localize and describe this object. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: localize and describe this object in the given 3D scene. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, describe this object first, then localize it. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
        dict(
            instruction='### human: describe then localize the object from the 3D scene. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
    ],
    'region_caption':[
        dict(
            instruction='### human: Describe the position of this object in relation to the surrounding objects in the 3D scene. ### assistant:',
            answer='{caption}',
            do_localize=False
        ),
        dict(
            instruction='### human: given the 3D scene, localize and describe the position of this object in relation to the surrounding objects in the 3D scene. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
    ]
    
}

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

def _rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        
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
            
            # Augment the scene data using shift and rotation
            self._all_scans_datasets = self._all_scans_datasets * 10
            self._all_region = self._all_region[:100000]
            if config.subset == 'train':
                self._all_scans_datasets = self._all_scans_datasets[:-10000]
                self._all_region = self._all_region[:-10000]
                self.objaverse_data.obj_ids = self.objaverse_data.obj_ids[:100000]
            else:
                self._all_scans_datasets = self._all_scans_datasets[-10000:]
                self._all_region = self._all_region[-10000:]
                self.objaverse_data.obj_ids = self.objaverse_data.obj_ids[-10000:]           
            
            
            # self.order_episodes = []
            # self.order_levels = []
            # self.order_episodes.extend(self._all_scans_datasets)
            # self.order_levels.extend(['scene'] * len(self._all_scans_datasets))
            # self.order_episodes.extend(self._all_region)
            # self.order_levels.extend(['region'] * len(self._all_region))
            # self.order_episodes.extend(self.objaverse_data.obj_ids)
            # self.order_levels.extend(['instance'] * len(self.objaverse_data.obj_ids))          
            # new_order_episodes = []
            # for lev, ep in zip(self.order_levels, self.order_episodes):
            #     new_order_episodes.append((ep, lev))
            # self.order_episodes = new_order_episodes
            
        
            # As diffent dataset has different number of points, we need to specify the dataset squence order 
            # to make sure samples from on batch come from the same level dataset
            batch_size_pre_rank = config.tbs
            self.order_episodes = []
            self.order_levels = []
            random.shuffle(self._all_scans_datasets)
            random.shuffle(self._all_region)
            random.shuffle(self.objaverse_data.obj_ids)
            if dist.is_initialized():
                dist.broadcast_object_list(self._all_scans_datasets, src=0)
                dist.broadcast_object_list(self._all_region, src=0)
                dist.broadcast_object_list(self.objaverse_data.obj_ids, src=0)
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
            
            self.order_episodes = self.order_episodes[::-1]
            self.order_levels = self.order_levels[::-1]
            
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
            points = _augment_pointcloud(points)
            points = self.pc_norm(points)
            # concat color
            # points = np.concatenate([points, colors/255], axis=1)
            return f'{scan_name}@{level}', dataset_name, (points.astype(np.float32), self._num_groups, self._group_size)
        elif level == 'region':
            region_data = np.load(os.path.join('data/SceneVerse/RegionAugData', data))
            dataset_name = data.split('/')[-1].split('_')[0]
            scan_name = data.split('/')[-1].split('_')[1]
            points, colors, instance_labels = region_data['points'], region_data['colors'], region_data['instance_labels']
            points, colors, instance_labels = self.down_sample(points, colors, instance_labels, self._region_npoint)
            points = _augment_pointcloud(points)
            points = self.pc_norm(points)
            # points = self._padding_pointcloud(points)
            # concat color
            # points = np.concatenate([points, colors/255], axis=1)
            return f'{scan_name}@{level}', dataset_name, (points.astype(np.float32), self._region_num_groups, self._region_group_size)
        elif level == 'instance':
            obj_pcd = self.objaverse_data.load_obj_pcd(data)
            points = obj_pcd[:, :3]
            colors = obj_pcd[:, 3:]
            points, colors, _ = self.down_sample(points, colors, npoint=self._instance_npoint)
            points = _augment_pointcloud(points)
            points = self.pc_norm(points)
            # points = self._padding_pointcloud(points)
            # concat color
            # points = np.concatenate([points, colors/255], axis=1)
            return f'{data}@object', 'Objaverse', (points.astype(np.float32), self._instance_num_groups, self._instance_group_size)
            

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
                
                # indices_within_bbox = []
                # for i, point in enumerate(points):
                #     if np.all(min_bound <= point) and np.all(point <= max_bound):
                #         indices_within_bbox.append(i)
                indices_within_bbox = np.where(np.all((points >= min_bound) & (points <= max_bound), axis=1))[0]
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
                    points=region_points, colors=region_colors, instance_labels=region_instance_labels, inst_to_label=inst_to_label)
                
                self._all_region.append((region_points, region_colors, region_instance_labels))
    
    def __len__(self):
        return len(self._all_region)
    
    def __getitem__(self, index):
        region_points, region_colors, region_instance_labels = self._all_region[index]
        
        points, colors, instance_labels = self.down_sample(points, colors, instance_labels)
        # region_points = self.pc_norm(region_points)
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

def padding_pointcloud(points):
    PAD_DATA_NUM = 40000
    pad_num = PAD_DATA_NUM - points.shape[0]
    if pad_num > 0:
        pad_points = np.zeros((pad_num, points.shape[-1])).astype(points.dtype)
        points = np.concatenate([points, pad_points], axis=0)
    return points
    
def augment_pointcloud(points):
    if np.random.random() > 0.5:
    # Flipping along the YZ plane
        points[:, 0] = -1 * points[:, 0]
    if np.random.random() > 0.5:
        # Flipping along the XZ plane
        points[:, 1] = -1 * points[:, 1]

    # Rotation along up-axis/Z-axis
    rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
    rot_mat = _rotz(rot_angle)
    points = np.dot(points, np.transpose(rot_mat))
    
    return points

def down_sample(points, colors, instance_labels=None, featrues=None, npoint=None):
    pcd_idxs = np.random.choice(len(points), size=npoint, replace=len(points) < npoint)
    points = points[pcd_idxs]
    colors = colors[pcd_idxs]
    instance_labels = instance_labels[pcd_idxs] if not instance_labels is None else None
    featrues = featrues[pcd_idxs] if not featrues is None else None
    return points, colors, instance_labels, featrues

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def shift_scale_points(pred_xyz, src_range, dst_range=None):
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

def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
    box_size = np.array([xmax-xmin, ymax-ymin, zmax-zmin])
    return center, box_size

def encode_box_coords(gt_box_centers_normalized, gt_box_sizes_normalized):
    grid_size_3d = 255
    # BOX_FORMAT = '<obj><loc{}><loc{}><loc{}><whl{}><whl{}><whl{}></obj>'
    BOX_FORMAT = '<obj>{},{},{},{},{},{}</obj>'
    center_normalized = gt_box_centers_normalized
    size_normalized = gt_box_sizes_normalized
    box_normalized = np.hstack((center_normalized, size_normalized))    # (-1, 6)
    # <cx, cy, cz, w, h, l>
    box_normalized = (box_normalized * grid_size_3d).astype(np.int64)
    return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)

def scale_points(pred_xyz, mult_factor):
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    scaled_xyz = pred_xyz * mult_factor[:, None, :]
    return scaled_xyz

def convert_objectpoints_to_bbox_str(points, object_points):
    center, whl = convert_pc_to_box(object_points)
    point_cloud_dims_min = points.min(axis=0)
    point_cloud_dims_max = points.max(axis=0)
    box_centers = center.astype(np.float32)
    center_normalizing_range = [
        np.zeros((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    ]
    box_centers_normalized = shift_scale_points(
        box_centers[None, ...],
        src_range=[
            point_cloud_dims_min[None, ...],
            point_cloud_dims_max[None, ...],
        ],
        dst_range=center_normalizing_range,
    )
    mult_factor = point_cloud_dims_max - point_cloud_dims_min
    box_sizes_normalized = scale_points(
        whl.astype(np.float32)[None, ...],
        mult_factor=1.0 / mult_factor[None, ...],
    )
    boxes_str = encode_box_coords(box_centers_normalized[0], box_sizes_normalized[0])
    return boxes_str


@DATASETS.register_module()
class SceneVerseLLMPretrainDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf', add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        
        self.SCENE_TOKEN = '<scene><scene_placehold></scene>'
        self.VP_TOKEN = '<vp_placehold>'
        
        special_tokens = ['<vp_placehold>', '<scene>', '<scene_placehold>', '</scene>', '<obj>', '</obj>']
        # xyz_prompt = '<loc{}>'
        # for i in range(255):
        #     special_tokens.append(xyz_prompt.format(i))
        # whl_prompt = '<whl{}>'
        # for i in range(255):
        #     special_tokens.append(whl_prompt.format(i))
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
        
        # If use openscene as encoder
        self.OPENSCENE = config.OPENSCENE
        
        # Load scene level data
        self._npoint = config.N_POINTS
        self._group_size = config.GROUP_SIZE
        self._num_groups = config.NUM_GROUP
        _all_dataset_name = ['3RScan', 'HM3D', 'MultiScan', 'ARKitScenes', 'ScanNet']
    
        self._all_dataset_root = 'data/SceneVerse'
        self._openscene_root = 'data/SceneVerse/OpenScene_Scan_Features'
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
        
        # Preprocess filter region caption that has instance label more than 1 in the caption ~83k
        instance_filter_relation_caption = json.load(open('data/SceneVerse/valid_region_relation.json', 'r'))
        exist_relation_caption = {'{}_{}.pth_{}.npz'.format(anno['dataset_name'],anno['scan_name'],anno['anno']["target_id"]):[] for anno in instance_filter_relation_caption}
        
        region_aug_data_dir = 'data/SceneVerse/RegionAugData'
        region_aug_pcd = os.listdir(region_aug_data_dir)
        # As dict search key will fast
        region_aug_pcd = {p:[] for p in region_aug_pcd}
        # valid_region_relation_path = []
        for dataset_name, annotations in all_dataset_dict.items():
            
            # Process scene caption: {scan_name: [captions]}
            # ~41k
            scene_annos = annotations.get('scene_caption')
            for scan_name, captions in scene_annos.items():
                self.all_scene_caption.extend([{'dataset_name':dataset_name, 
                                                "scan_name":scan_name, 
                                                "anno":{'utterance':cap}, 
                                                "task_name": "scene_caption",
                                                'episode_id':'{}#{}#{}'.format(dataset_name, scan_name, ci)} 
                                               for ci,cap in enumerate(captions['captions'])])
            
            # Process region caption: [dict_keys(['item_id', 'scan_id', 'target_id', 'instance_type', 'utterance']), xxx]  
            
            # Further add some relation caption that has not instance label for augment ~140k
            region_annos = annotations.get('relation_caption')
            for ra in (region_annos):
                scan_name = ra['scan_id']
                path = f'{dataset_name}_{scan_name}.pth_{ra["target_id"]}.npz'
                # Check if the pcd corresponding to the region caption exists
                if path not in region_aug_pcd.keys() or path not in exist_relation_caption.keys():
                    continue
                self.all_relation_caption.append({'dataset_name':dataset_name, 
                                                  "scan_name":scan_name, 
                                                  "anno":ra, 
                                                  "task_name": "region_caption",
                                                  'episode_id':'{}#{}#{}'.format(dataset_name, scan_name, ra['utterance'])})
                
                ''' Code for filter region caption that has instance label more than 1 in the caption
                region_data = np.load(os.path.join('data/SceneVerse/RegionAugData1', path), allow_pickle=True)
                points, colors, instance_labels, label2text = region_data['points'], region_data['colors'], region_data['instance_labels'], region_data['inst_to_label']
                caption = ra['utterance']
                unique_label = np.unique(instance_labels)
                label2text = label2text.item()
                unique_text = [label2text[int(l)] for l in unique_label if not l <= 0]
                text_in_caption = []
                position = []
                for text in unique_text:
                    pos = caption.find(text)
                    if not pos == -1:
                        text_in_caption.append(text)
                        position.append(pos)
                
                if len(position) == len(np.unique(position)) and len(text_in_caption) >= 2:
                    text_in_caption = [(t,p) for t, p in zip(text_in_caption, position)]
                    valid_region_relation_path.append(path)
                    self.all_relation_caption.append({'dataset_name':dataset_name, "scan_name":scan_name, "anno":ra, "task_name": "region_caption", 'text_in_caption': text_in_caption})
                '''
            
            # Process object caption: [dict_keys(['item_id', 'scan_id', 'target_id', 'instance_type', 'utterance']), xxx]
            # ~136k
            
            ## As HM3D only contains annotations from template like 'the pointcloud of xxx'
            if dataset_name == 'HM3D':
                continue
            
            ## Openscene as encoder only can use scene object caption data
            if self.OPENSCENE:
                object_annos = annotations.get('object_caption')
                for oa in object_annos:
                    scan_name = oa['scan_id']
                    # As some object instance pointcloud will miss after downsample, we only keep the object caption with pcd data large than threshold
                    if f'{dataset_name}_{scan_name}.pth_{oa["target_id"]}.npz' not in region_aug_pcd.keys():
                        continue
                    self.all_object_caption.append({'dataset_name':dataset_name, 
                                                    "scan_name":scan_name, 
                                                    "anno":oa, 
                                                    "task_name": "object_caption",
                                                    'episode_id':'{}#{}#{}'.format(dataset_name, scan_name, oa['utterance'])})

        # with open('data/SceneVerse/valid_region_relation_path.json', 'w') as f:
        #     json.dump(valid_region_relation_path, f)
        # assert False
        
        # Load region level data
        self._region_npoint = config.REGION_N_POINTS
        self._region_group_size = config.REGION_GROUP_SIZE
        self._region_num_groups = config.REGION_NUM_GROUP
        
        ''' Augment object caption task into object grouding task
        all_grouding_object = copy.deepcopy(self.all_object_caption)
        for data in all_grouding_object:
            data['task_name'] = 'object_grouding'
        self.all_scene_caption.extend(all_grouding_object)
        '''
        
        # Load some objaverse caption data 
        self._instance_npoint = config.INSTANCE_N_POINTS
        self._instance_group_size = config.INSTANCE_GROUP_SIZE
        self._instance_num_groups = config.INSTANCE_NUM_GROUP
        
        # Openscene cannet deal with only object input 
        if not self.OPENSCENE:
            self._load_objaverse_data()
            objaverse_objs = list(self.objaverse_data.obj_cap_dict.keys())
            random.shuffle(objaverse_objs)
            # dist.broadcast_object_list(objaverse_objs, src=0)
            for obj_dict in objaverse_objs:
                self.all_object_caption.append({'dataset_name':'Objaverse', 
                                                "scan_name":obj_dict, 
                                                "task_name": "object_caption",
                                                "anno":{
                                                        'item_id': obj_dict,
                                                        'scan_id': obj_dict,
                                                        'utterance': self.objaverse_data.obj_cap_dict[obj_dict]
                                                        },
                                                'episode_id': obj_dict})
        
        # Load some LEO scene caption data 
        leo_scene_train_caption = json.load(open('data/LEO_DATA/annotations/alignment/scene_caption/3rscan_scenecap_train.json', 'r'))
        leo_scene_val_caption = json.load(open('data/LEO_DATA/annotations/alignment/scene_caption/3rscan_scenecap_val.json', 'r'))
        extend_leo_scene_caption = []
        for scene_name, v in leo_scene_train_caption.items():
            for ci,cap in enumerate(v):
                extend_leo_scene_caption.append({
                                            'dataset_name':'3RScan', 
                                            "scan_name":scene_name, 
                                            "task_name": "scene_caption",
                                            "anno":{
                                                    'utterance': cap['response']
                                                },
                                            'episode_id':'leo#3RScan#{}#{}'.format(scene_name, cap['response'])
                                            })
        for scene_name, v in leo_scene_val_caption.items():
            for ci,cap in enumerate(v):
                extend_leo_scene_caption.append({
                                            'dataset_name':'3RScan', 
                                            "scan_name":scene_name, 
                                            "task_name": "scene_caption",
                                            "anno":{
                                                    'utterance': cap['response']
                                                },
                                            'episode_id':'leo#3RScan#{}#{}'.format(scene_name, cap['response'])
                                            })
                
        self.all_scene_caption.extend(extend_leo_scene_caption)
        
        # Load some grouding scene caption data ~107k from GroundLLM https://arxiv.org/pdf/2405.10370
        # grounding_scene_caption_anno = json.load(open('data/SceneVerse/ScanNet/annotations/scene_caption/groundedscenecaption_format.json', 'r'))
        # for gsc in grounding_scene_caption_anno:
        #     self.all_scene_caption.append({'dataset_name':'ScanNet', 
        #                                     "scan_name":gsc['scene_id'], 
        #                                     "task_name": "scene_caption",
        #                                     "anno":{
        #                                             'utterance': gsc['description'],
        #                                             'all_phrases_positions': gsc['all_phrases_positions'],
        #                                             'object_ids': gsc['object_ids'],
        #                                         }
        #                                     })

        # random.seed(0)
        # self.all_relation_caption.extend(instance_filter_relation_caption)
        random.shuffle(self.all_scene_caption)
        random.shuffle(self.all_relation_caption)
        random.shuffle(self.all_object_caption)
        
        dist.broadcast_object_list(self.all_scene_caption, src=0)
        dist.broadcast_object_list(self.all_relation_caption, src=0)
        dist.broadcast_object_list(self.all_object_caption, src=0)
        
        self.all_relation_caption  = self.all_relation_caption[:len(self.all_scene_caption)]
        self.all_object_caption  = self.all_object_caption[:len(self.all_scene_caption)]
        
        if config.subset == 'train':
            self.all_scene_caption = self.all_scene_caption[:-2000]
            self.all_relation_caption = self.all_relation_caption[:-2000]
            self.all_object_caption = self.all_object_caption[:-2000]
        else:
            self.all_scene_caption = self.all_scene_caption[-2000:]
            self.all_relation_caption = self.all_relation_caption[-2000:]
            self.all_object_caption = self.all_object_caption[-2000:]

        print_log(f'[DATASET] {len(self.all_scene_caption)} scene captions were loaded from scan data', logger = 'SceneVerse')
        print_log(f'[DATASET] {len(self.all_relation_caption)} relation captions were loaded from scan data', logger = 'SceneVerse')
        print_log(f'[DATASET] {len(self.all_object_caption)} object captions were loaded from scan data and objaverse', logger = 'SceneVerse')
        
        # Prepare corpus for evaluation
        self.corpus = {
            'scene_caption': copy.deepcopy(self.all_scene_caption),
            'object_caption': copy.deepcopy(self.all_object_caption),
            'region_caption': copy.deepcopy(self.all_relation_caption)
        }
        
        self.order_episodes = []
        self.order_levels = []
        
        # Shuffle code
        # self.order_episodes.extend(self.all_scene_caption)
        # self.order_levels.extend(['scene'] * len(self.all_scene_caption))
        self.order_episodes.extend(self.all_relation_caption)
        self.order_levels.extend(['region'] * len(self.all_relation_caption))
        self.order_episodes.extend(self.all_object_caption)
        self.order_levels.extend(['instance'] * len(self.all_object_caption))       
        
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
        if not self.OPENSCENE:
            dataset_root = os.path.join(self._all_dataset_root, dataset_name)
            annotation_root = os.path.join(dataset_root, 'annotations')
            scan_data_root = os.path.join(dataset_root, 'scan_data')
            
            inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
            pcd_path = os.path.join(scan_data_root,'pcd_with_global_alignment')

            points, colors, pcds, instance_labels, inst_to_label = self._load_scan(pcd_path, inst2label_path, scan_name)
            features = None
        else:
            dataset_root = os.path.join(self._all_dataset_root, dataset_name)
            scan_data_root = os.path.join(dataset_root, 'scan_data')
            inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
            inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}")) 
            dataset_root = os.path.join(self._openscene_root, dataset_name)
            dict = torch.load(os.path.join(dataset_root, scan_name), map_location='cpu')
            points = dict['points'].numpy().astype(np.float32)
            colors = dict['colors'].numpy()
            features = dict['features'].numpy().astype(np.float32)
            instance_labels = dict['instance_labels'].numpy()
        return points, colors, features, instance_labels, inst_to_label

    def __len__(self):
        return len(self.order_episodes)
    
    def __getitem__(self, index):
        
        level = self.order_levels[index]
        data = self.order_episodes[index]
        dataset_name, scan_name, anno, task_name = data['dataset_name'], data['scan_name'], data['anno'], data['task_name']
        
        self.tokenizer_config = dict(
            max_length=256, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        
        points, colors, features, instance_labels, inst_to_label = self._load_scan_data(f'{scan_name}.pth', dataset_name)
        points = pc_norm(points)
        click_query = np.zeros((1, 3))
        click_mask = np.zeros((1,))
        box_mask = np.zeros((1,))
        box_query = np.zeros((features.shape[-1],))
        
        points, colors, instance_labels, features = down_sample(points, colors, instance_labels, features, npoint=self._npoint)
        
        ret_dict = {
            'num_groups': self._num_groups,
            'group_size': self._group_size,
            'dataset_name': dataset_name,
            'level': 'scene',
            'scan_name': scan_name,
            'task_name': task_name,
            'episode_id': data['episode_id'],
            'click_query': click_query.astype(np.float32),
            'click_mask': click_mask.astype(np.float32),
            'box_mask': box_mask.astype(np.float32),
            'box_query': box_query.astype(np.float32),
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
        }
        if self.OPENSCENE:
            ret_dict['features'] = features
        
        if level == 'scene':
            intruction = '### human: describe the 3D scene. ### assistant:'
            intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            answers = anno['utterance']
            llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
            )
            
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)

            return ret_dict
            
        elif level == 'region':
            instance_id = int(anno['target_id'])
            
            if random.random() < 0.5 or self.config.subset == 'val':
                click_query[0] = random.choice(points[instance_labels == instance_id])
                click_mask[0] = 1
            else:
                box_mask[0] = 1
                box_query = features[instance_labels == instance_id].mean(0)
            
            ret_dict.update({
                'click_query': click_query.astype(np.float32),
                'click_mask': click_mask.astype(np.float32),
                'box_mask': box_mask.astype(np.float32),
                'box_query': box_query.astype(np.float32),
            })
            
            if self.config.subset == 'train':
                prompt = deepcopy(random.choice(TASK_PROMPT[task_name]))
            else:
                prompt = deepcopy(TASK_PROMPT[task_name][0])
            boxes = convert_objectpoints_to_bbox_str(points, points[instance_labels == instance_id])
            caption = anno['utterance']
            answers = prompt['answer'].format(locations=boxes, caption=caption)
            intruction = random.choice(TASK_PROMPT[task_name])['instruction']
            intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
            )
            
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
            
            return ret_dict
        
        elif level == 'instance':
            instance_id = int(anno['target_id'])
                
            if random.random() < 0.5 or self.config.subset == 'val':
                click_query[0] = random.choice(points[instance_labels == instance_id])
                click_mask[0] = 1
            else:
                box_mask[0] = 1
                box_query = features[instance_labels == instance_id].mean(0)
            ret_dict.update({
                'click_query': click_query.astype(np.float32),
                'click_mask': click_mask.astype(np.float32),
                'box_mask': box_mask.astype(np.float32),
                'box_query': box_query.astype(np.float32),
            })
            
            boxes = convert_objectpoints_to_bbox_str(points, points[instance_labels == instance_id])
            if self.config.subset == 'train':
                prompt = deepcopy(random.choice(TASK_PROMPT[task_name]))
            else:
                prompt = deepcopy(TASK_PROMPT[task_name][0])
                
            intruction = prompt['instruction']
            intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            caption = anno['utterance']
            answers = prompt['answer'].format(locations=boxes, caption=caption)
            llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
            )
            
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)

            return ret_dict


def proc_object_tokens_for_objectcentric(points, instance_labels, sparse_obj_token_label=None, tgt_label=None):
        object_tokens = []
        center = []
        room_label_offset = 10000
        _num_groups = 128
        obj_pcd_num = 256
        obj_token_label = []
        obj_loc = []
        if sparse_obj_token_label is None:
            if tgt_label is not None:
                mask = instance_labels == tgt_label
                object_points = points[mask]
                center.append(object_points[:, :3].mean(0))
                obj_center = object_points[:, :3].mean(0)
                obj_size = object_points[:, :3].max(0) - object_points[:, :3].min(0)
                obj_loc.append(np.concatenate([obj_center, obj_size], 0))
                idx = np.random.choice(len(object_points), obj_pcd_num, replace=True)
                object_points = object_points[idx]
                object_tokens.append(object_points)
                obj_token_label.append(tgt_label)
            else:
                for label in np.unique(instance_labels):
                    if label <= 0 or label == room_label_offset or label == room_label_offset*2 or label == room_label_offset*3:
                        continue
                    mask = instance_labels == label
                    object_points = points[mask]
                    center.append(object_points[:, :3].mean(0))
                    obj_center = object_points[:, :3].mean(0)
                    obj_size = object_points[:, :3].max(0) - object_points[:, :3].min(0)
                    obj_loc.append(np.concatenate([obj_center, obj_size], 0))
                    idx = np.random.choice(len(object_points), obj_pcd_num, replace=True)
                    object_points = object_points[idx]
                    object_tokens.append(object_points)
                    obj_token_label.append(label)
        # else:
        #     # As dense vision token needs be the same order as the sparse token
        #     for label in sparse_obj_token_label:
        #         mask = instance_labels == label
        #         num = 2
        #         if mask.sum() == 0:
        #             center.append(np.zeros((3,)).astype(np.float32))
        #             object_tokens.append(np.zeros((num, 768)).astype(np.float32))
        #         else:
        #             object_points = points[mask]
        #             object_features = features[mask]
        #             dim = 0
        #             usable_size = (object_features.shape[dim] // num) * num
        #             object_features = object_features[:usable_size, :]
        #             splits = np.split(object_features, num, axis=dim)
        #             means = [np.mean(split, axis=dim, keepdims=True) for split in splits]
        #             object_features = np.concatenate(means, axis=dim)
        #             center.append(object_points.mean(0))
        #             object_tokens.append(object_features)
        #         obj_token_label.append(label)

        center = np.array(center)
        obj_loc = np.array(obj_loc)
        object_tokens = np.array(object_tokens)
        if len(object_tokens) <= _num_groups:
            # pad to group size
            pad_size = _num_groups - len(object_tokens)
            object_mask = np.concatenate([np.ones(len(object_tokens)), np.zeros(pad_size)], 0)
            if sparse_obj_token_label is None:
                object_tokens = np.concatenate([object_tokens, np.zeros((pad_size, object_tokens.shape[-2], object_tokens.shape[-1]))], 0)
            # else:
            #     object_tokens = np.concatenate([object_tokens, np.zeros((pad_size, num, object_tokens.shape[-1]))], 0)
            center = np.concatenate([center, np.zeros((pad_size, center.shape[-1]))], 0)
            obj_loc = np.concatenate([obj_loc, np.zeros((pad_size, obj_loc.shape[-1]))], 0)
            obj_token_label.extend([0] * pad_size)
        else:
            object_mask = np.ones(len(object_tokens))
            object_tokens = object_tokens[:_num_groups]
            center = center[:_num_groups]
            object_mask = object_mask[:_num_groups]
            obj_loc = obj_loc[:_num_groups]
            obj_token_label = obj_token_label[:_num_groups]

        return object_tokens, center, object_mask, obj_loc, obj_token_label
 
        
@DATASETS.register_module()
class SceneVerseLLMFinetuneDataset(Dataset):
    '''
        Downstream datasets
            1. QA 
                ScanQA & 3RScanQA(LEO) ~84k
            2. Object QA
                Object caption & Grouding object (SceneVerse) ~132k
            3. Scene understanding
                Scene caption ~1k & scene dia ~8k & embodied QA ~15k & embodied plan ~16k (3dllm) ~40k
            4. Text
                shareGPT ~40k
    '''
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf', add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.qtokenizer = AutoTokenizer.from_pretrained('ckpts/bert-base-uncased')
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'
        
        self.wohd = config.get('wohd', False)
        self.config = config
        
        if not hasattr(self.config, 'differ_prompt'):
            self.config.differ_prompt = False
        
        self.USE_OBJECTCENTRIC = config.get('USE_OBJECTCENTRIC', False)
        self.OBJECT_CAPTION_WO_VP = config.get('OBJECT_CAPTION_WO_VP', False)
        self.room_label_offset = 10000
        
        self.SCENE_TOKEN = '<scene><scene_placehold></scene>'
        self.VP_TOKEN = '<vp_placehold>'
        
        special_tokens = ['<vp_placehold>', '<scene>', '<scene_placehold>', '</scene>', '<obj>', '</obj>']
        # xyz_prompt = '<loc{}>'
        # for i in range(255):
        #     special_tokens.append(xyz_prompt.format(i))
        # whl_prompt = '<whl{}>'
        # for i in range(255):
        #     special_tokens.append(whl_prompt.format(i))
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
        
        # If use openscene as encoder
        self.OPENSCENE = config.OPENSCENE
        self._openscene_root = 'data/SceneVerse/OpenScene_Scan_Features'
        
        # If use extend dataset for finetune instead of only ll3da dataset
        self.EXTEND = config.get('EXTEND', False)
        if self.EXTEND:
            print_log("Use extend dataset", logger = 'SceneVerse')
        self.UES_HD_DATA = config.get('UES_HD_DATA', False)
        if self.UES_HD_DATA:
            print_log("Use HD dataset", logger = 'SceneVerse')
        
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
            
        # Load part 1 task: QA
        self.all_scene_qa = []
        self.scanqa_anno = json.load(open(f'data/SceneVerse/ScanNet/annotations/qa/ScanQA_v1.0_{config.subset}.json'))
        for si,scene_cap in enumerate(self.scanqa_anno):
            scan_name = scene_cap['scene_id']
            self.all_scene_qa.append({'dataset_name':'ScanNet', 
                                      "scan_name":scan_name, 
                                      "anno":scene_cap, 
                                      "task_name": "scene_qa",
                                      'episode_id':'{}#{}#{}'.format(dataset_name, scan_name, scene_cap['question'])
                                      })
        
        ## Only in train
        if config.subset == 'train' and self.EXTEND:
            self.rscanqa_anno = json.load(open(f'data/LEO_DATA/annotations/instruction/3rscanqa/3rscan_qa_{config.subset}.json'))
            ## Transfer rscan qa format to scan qa format
            tmp_rscanqa_anno = []
            for scene_id, qas in self.rscanqa_anno.items():
                for qa in qas['response']:
                    tmp_rscanqa_anno.append({
                        'scene_id': scene_id,
                        'question': qa['Q'],
                        'object_names': qa.get('T', 'None'),
                        'answers': qa['A'],
                    })
            self.rscanqa_anno = tmp_rscanqa_anno
            for scene_cap in self.rscanqa_anno:
                scan_name = scene_cap['scene_id']
                self.all_scene_qa.append({'dataset_name':'3RScan', 
                                          "scan_name":scan_name, 
                                          "anno":scene_cap, 
                                          "task_name": "scene_qa",
                                          'episode_id':'leo#{}#{}#{}'.format(dataset_name, scan_name, scene_cap['question'])})
        
        # Load part 2 task: object caption and object grounding from all scans of sceneverse
        ## Note that we only keep the scannet part for eval
        # region_aug_data_dir = 'data/SceneVerse/RegionAugData'
        # region_aug_pcd = os.listdir(region_aug_data_dir)
        # region_aug_pcd = {p:[] for p in region_aug_pcd}
        self.all_object_caption = []
        object_caption_anno_name = 'ssg_obj_caption_gpt.json'
        for dn in _all_dataset_name:
            if dn == 'ScanNet':
                tmp_object_caption = json.load(open(f'data/SceneVerse/ScanNet/annotations/object_caption/ScanRefer_filtered_{config.subset}_qa_format.json'))
                for oc in tmp_object_caption:
                    if config.subset == 'train':
                        for ans in oc['answers']:
                            scan_name = oc['scene_id']
                            anno = {'scene_id':oc['scene_id'], 'target_id':oc['object_id'], 'answers':[ans], 'object_name':oc['object_name']}
                            self.all_object_caption.append({'dataset_name':dn, 
                                                            "scan_name":scan_name, 
                                                            "anno":anno, 
                                                            "task_name": "object_caption",
                                                            'episode_id':'{}#{}#{}#{}'.format(dataset_name, scan_name, oc['object_id'], ans)
                                                            })
                    ## In eval we hava multiple answers
                    else:
                        scan_name = oc['scene_id']
                        self.all_object_caption.append({'dataset_name':dn, 
                                                        "scan_name":scan_name, 
                                                        "anno":oc, 
                                                        "task_name": "object_caption",
                                                        'episode_id':'{}#{}#{}'.format(dataset_name, scan_name, oc['object_id'])
                                                        })
            # As HM3D only contains annotations from template like 'the pointcloud of xxx'
            elif config.subset == 'train' and not dn == 'HM3D' and self.EXTEND:
                tmp_object_caption = json.load(open(f'data/SceneVerse/{dn}/annotations/{object_caption_anno_name}'))
                for oc in tmp_object_caption:
                    scan_name = oc['scan_id']
                    # As some object instance pointcloud will miss after downsample, we only keep the object caption with pcd data large than threshold
                    if f'{dn}_{scan_name}.pth_{oc["target_id"]}.npz' not in region_aug_pcd.keys():
                        continue
                    self.all_object_caption.append({'dataset_name':dn, 
                                                    "scan_name":scan_name, 
                                                    "anno":oc, 
                                                    "task_name": "object_caption",
                                                    'episode_id':'{}#{}#{}'.format(dataset_name, scan_name, oc['target_id'])})

        # Load part 3 task: scene understanding from all scans of sceneverse
        ## All in scene_id question answer format
        self.all_scene_understanding = []
        ## Only in train
        if config.subset == 'train':
            embodied_dialogue_anno = json.load(open(f'data/SceneVerse/3D_LLM/3d_llm_embodied_dialogue_filtered_{config.subset}.json'))
            for ed in embodied_dialogue_anno:
                scan_name = ed['scene_id']
                self.all_scene_understanding.append({'dataset_name':'ScanNet', 
                                                    "scan_name":scan_name, 
                                                    "anno":ed, 
                                                    "task_name": "scene_understanding",
                                                    'episode_id': 'none'
                                                    })
            embodied_planning_anno = json.load(open(f'data/SceneVerse/3D_LLM/3d_llm_embodied_planning_filtered_{config.subset}.json'))
            for ep in embodied_planning_anno:
                scan_name = ep['scene_id']
                self.all_scene_understanding.append({'dataset_name':'ScanNet', 
                                                    "scan_name":scan_name, 
                                                    "anno":ep, 
                                                    "task_name": "scene_understanding",
                                                    'episode_id': 'none'
                                                    })
            embodied_question_answer_anno = json.load(open(f'data/SceneVerse/3D_LLM/3d_llm_embodied_question_answer_{config.subset}.json'))
            for eqa in embodied_question_answer_anno:
                scan_name = eqa['scene_id']                
                self.all_scene_understanding.append({'dataset_name':'ScanNet', 
                                                    "scan_name":scan_name,
                                                    "anno":eqa, 
                                                    "task_name": "scene_understanding",
                                                    'episode_id': 'none'
                                                    })
            
            if self.EXTEND:
                scene_caption_anno_name = 'scene_cap.json'
                for dn in _all_dataset_name:
                    if not dn == 'ScanNet':
                        tmp_scene_caption = json.load(open(f'data/SceneVerse/{dn}/annotations/{scene_caption_anno_name}'))
                        for scan_name, captions in tmp_scene_caption.items():
                            self.all_scene_understanding.extend([{'dataset_name':dn, 
                                                            "scan_name":scan_name, 
                                                            "anno":{
                                                                'answers': [cap],
                                                                'question': '### human: Describe the room. ### assistant:',
                                                                }, 
                                                            
                                                            "task_name": "scene_understanding",
                                                            'episode_id':'sceneverse#{}#{}#{}'.format(dataset_name, scan_name, ci)} 
                                                        for ci,cap in enumerate(captions['captions'])])
        
        if self.EXTEND:
            # Load some LEO scene caption data 
            leo_scene_train_caption = json.load(open('data/LEO_DATA/annotations/alignment/scene_caption/3rscan_scenecap_train.json', 'r'))
            leo_scene_val_caption = json.load(open('data/LEO_DATA/annotations/alignment/scene_caption/3rscan_scenecap_val.json', 'r'))
            extend_leo_scene_caption = []
            for scene_name, v in leo_scene_train_caption.items():
                for ci,cap in enumerate(v):
                    extend_leo_scene_caption.append({
                                                'dataset_name':'3RScan', 
                                                "scan_name":scene_name, 
                                                "task_name": "scene_understanding",
                                                "anno":{
                                                    'question': '### human: Describe the room. ### assistant:',
                                                    'answers': [cap['response']]
                                                    },
                                                'episode_id':'leo#{}#{}#{}'.format(dataset_name, scene_name, cap['response'])
                                                })
            for scene_name, v in leo_scene_val_caption.items():
                for ci,cap in enumerate(v):
                    extend_leo_scene_caption.append({
                                                'dataset_name':'3RScan', 
                                                "scan_name":scene_name, 
                                                "task_name": "scene_understanding",
                                                "anno":{
                                                        'question': '### human: Describe the room. ### assistant:',
                                                        'answers': [cap['response']]
                                                    },
                                                'episode_id':'leo#{}#{}#{}'.format(dataset_name, scene_name, cap['response'])
                                                })                
            
        scene_caption_anno = json.load(open(f'data/SceneVerse/3D_LLM/3d_llm_scene_description_{config.subset}.json'))
        for sc in scene_caption_anno:
            scan_name = sc['scene_id']
            self.all_scene_understanding.append({'dataset_name':'ScanNet', 
                                                "scan_name":scan_name, 
                                                "anno":sc, 
                                                "task_name": "scene_understanding",
                                                'episode_id':'{}#{}'.format(dataset_name, scan_name)
                                                })

        # Load HD QA datasets
        if self.UES_HD_DATA:
            hd_qa_source_dir_f = 'data/SceneVerse/HM3D/annotations/qa/qa_pairs/{}.json'.format(config.subset)
            with open(hd_qa_source_dir_f, 'r') as f:
                datas = json.load(f)
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
                        self.all_scene_qa.append({'dataset_name':'HM3D', 
                                                "scan_name":scan_name, 
                                                'instance_room_id': epi['scan_id'].split('_')[-1],
                                                "anno":anno, 
                                                "task_name": "hd_scene_qa",
                                                'region_id': qa['region_id'],
                                                'episode_id':'{}#{}#{}#{}'.format('HM3D', scan_name, qa['region_id'], qa['question'])
                                                })
        
        print_log(f'[DATASET] {len(self.all_scene_qa)} scene qa were loaded from scan data', logger = 'SceneVerse')
        print_log(f'[DATASET] {len(self.all_object_caption)} object captions were loaded from scan data', logger = 'SceneVerse')
        print_log(f'[DATASET] {len(self.all_scene_understanding)} scene captions were loaded from scan data', logger = 'SceneVerse')
    
        # Load region level data
        self._region_npoint = config.REGION_N_POINTS
        self._region_group_size = config.REGION_GROUP_SIZE
        self._region_num_groups = config.REGION_NUM_GROUP
        
        # Load some objaverse caption data 
        self._instance_npoint = config.INSTANCE_N_POINTS
        self._instance_group_size = config.INSTANCE_GROUP_SIZE
        self._instance_num_groups = config.INSTANCE_NUM_GROUP
        
        if not hasattr(self, 'all_object_grouding'):
            self.all_object_grouding = []
        
        # Prepare corpus for evaluation
        self.corpus = {
            'scene_qa': copy.deepcopy(self.all_scene_qa),
            'hd_scene_qa': copy.deepcopy(self.all_scene_qa),
            'object_caption': copy.deepcopy(self.all_object_caption),
            'scene_understanding': copy.deepcopy(self.all_scene_understanding)
        }
        
        self.order_episodes = []
        self.order_episodes.extend(self.all_scene_qa)
        if self.config.subset == 'train':
            self.order_episodes.extend(self.all_scene_understanding)
            self.order_episodes.extend(self.all_object_caption)

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
        if not self.OPENSCENE:
            dataset_root = os.path.join(self._all_dataset_root, dataset_name)
            annotation_root = os.path.join(dataset_root, 'annotations')
            scan_data_root = os.path.join(dataset_root, 'scan_data')
            
            inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
            pcd_path = os.path.join(scan_data_root,'pcd_with_global_alignment')

            points, colors, pcds, instance_labels, inst_to_label = self._load_scan(pcd_path, inst2label_path, scan_name)
            features = None
        else:
            dataset_root = os.path.join(self._all_dataset_root, dataset_name)
            scan_data_root = os.path.join(dataset_root, 'scan_data')
            inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
            inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}")) 
            dataset_root = os.path.join(self._openscene_root, dataset_name)
            dict = torch.load(os.path.join(dataset_root, scan_name), map_location='cpu')
            points = dict['points'].numpy().astype(np.float32)
            colors = dict['colors'].numpy()
            features = dict['features'].numpy().astype(np.float32)
            instance_labels = dict['instance_labels'].numpy()
        return points, colors, features, instance_labels, inst_to_label

    def __len__(self):
        return len(self.order_episodes)
    
    def __getitem__(self, index):
        
        data = self.order_episodes[index]
        dataset_name, scan_name, anno, task_name = data['dataset_name'], data['scan_name'], data['anno'], data['task_name']
        
        self.tokenizer_config = dict(
            max_length= 256, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        if task_name == 'hd_scene_qa':
            room_center = torch.load(os.path.join('data/SceneVerse/HM3D/scan_data/room_center', '{}.pth'.format(scan_name)))
            region_id = data['region_id']
            instance_room_id = data['instance_room_id']
            points = []
            colors = []
            features = []
            instance_labels = []
            inst_to_label = {}
            for ri, room_id in enumerate(region_id.split('-')):
                pts, cols, fts, ils, itl = self._load_scan_data(f'{scan_name}_{room_id}.pth', dataset_name)
                points.extend(pts + room_center[room_id]['center'])
                colors.extend(cols)
                features.extend(fts)
                if room_id == instance_room_id:
                    instance_labels.extend(ils)
                else:
                    if self.USE_OBJECTCENTRIC:
                        instance_labels.extend(ils+int(self.room_label_offset*ri))
                    else:
                        instance_labels.extend(np.zeros_like(ils).astype(ils.dtype))
                inst_to_label[room_id] = itl
            points = np.array(points)
            colors = np.array(colors)
            features = np.array(features)
            instance_labels = np.array(instance_labels)
        else:
            points, colors, features, instance_labels, inst_to_label = self._load_scan_data(f'{scan_name}.pth', dataset_name)
        points = pc_norm(points)
        
        click_query = np.zeros((1, 3))
        click_mask = np.zeros((1,))
        box_mask = np.zeros((1,))
        box_query = np.zeros((features.shape[-1],))
        
        # Get HD Info
        N = 160000
        if not self.wohd:
            hd_points, hd_features, hd_instance_labels, _ = down_sample(points, features, instance_labels, npoint=N)
            hd_points = hd_points.astype(np.float32)
            hd_features = hd_features.astype(np.float32)
        else:
            hd_points = None
            hd_features = None

        points, colors, instance_labels, features = down_sample(points, colors, instance_labels, features, npoint=self._npoint)
        pcds = np.concatenate([points, colors/255], -1)
        self.obj_pcd_num = 256 
        tgt_label = int(anno['target_id']) if self.OBJECT_CAPTION_WO_VP and task_name == 'object_caption' else None 
        obj_tokens, obj_center, obj_mask, obj_loc, obj_token_label = proc_object_tokens_for_objectcentric(pcds, instance_labels, tgt_label=tgt_label)
        # if not self.wohd:
        #     hd_obj_tokens, hd_obj_center, hd_obj_mask, _ = self.proc_object_tokens_for_objectcentric(hd_points, hd_features, hd_instance_labels, sparse_obj_token_label)
        # else:
        #     hd_obj_tokens = None
        #     hd_obj_center = None
        #     hd_obj_mask = None
        
        ret_dict = {
            'num_groups': self._num_groups,
            'group_size': self._group_size,
            'dataset_name': dataset_name,
            'level': 'scene',
            'scan_name': scan_name,
            'task_name': task_name,
            'episode_id': data['episode_id'],
            'click_query': click_query.astype(np.float32),
            'click_mask': click_mask.astype(np.float32),
            'box_mask': box_mask.astype(np.float32),
            'box_query': box_query.astype(np.float32),
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
        }

        if self.USE_OBJECTCENTRIC:
            ret_dict['obj_tokens'] = obj_tokens.astype(np.float32)
            ret_dict['obj_center'] = obj_center.astype(np.float32)
            ret_dict['obj_mask'] = obj_mask.astype(np.int64)
            ret_dict['obj_loc'] = obj_loc.astype(np.float32)
            ret_dict['obj_token_label'] = np.array(obj_token_label).astype(np.int64)
        
        if not self.wohd:
            # if self.USE_OBJECTCENTRIC:
            #     ret_dict['hd_obj_tokens'] = hd_obj_tokens.astype(np.float32)
            #     # ret_dict['hd_obj_center'] = hd_obj_center.astype(np.float32)
            #     # ret_dict['hd_obj_mask'] = hd_obj_mask.astype(np.float32)
            # else:
            ret_dict['hd_points'] = hd_points
            ret_dict['hd_features'] = hd_features
            ret_dict['hd_instance_labels'] = hd_instance_labels.astype(np.int64)
        
        if self.OPENSCENE:
            ret_dict['features'] = features
    
        if task_name == 'scene_qa':
            target_obj_id = None
            if self.config.subset == 'train' and random.random() < 0.25 and 'object_ids' in anno.keys():
                target_obj_id = random.choice(anno['object_ids'])
                object_points = points[instance_labels == target_obj_id]    # npt x 3
                click_query[0] = random.choice(object_points)
                click_mask[0] = 1
            # elif self.config.subset == 'val':
            #     target_obj_id = random.choice(anno['object_ids'])
            #     object_points = points[instance_labels == target_obj_id]    # npt x 3
            #     click_query[0] = random.choice(object_points)
            #     click_mask[0] = 1
            ret_dict.update({
                'click_query': click_query.astype(np.float32),
                'click_mask': click_mask.astype(np.float32),
                'box_mask': box_mask.astype(np.float32),
                'box_query': box_query.astype(np.float32),
            })
            
            question = anno['question']
            # build prompts
            if 'object_ids' in anno.keys():
                if self.config.subset == 'train' and len(anno['object_ids']) == 1 :
                    object_points = points[instance_labels == (random.choice(anno['object_ids']) if target_obj_id is None else target_obj_id)]    
                    boxes = convert_objectpoints_to_bbox_str(points, object_points)
                    prompt = deepcopy(random.choice(TASK_PROMPT[task_name]))
                else:
                    prompt = deepcopy(TASK_PROMPT[task_name][0])
                    boxes = '' 
            else:
                prompt = deepcopy(TASK_PROMPT[task_name][0])
                boxes = ''
            intruction = prompt['instruction'].format(locations=boxes, question=question)

            qformer_prompt = self.qtokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            ret_dict['qformer_instruction'] = qformer_prompt['input_ids'][0].astype(np.int64)
            ret_dict['qformer_instruction_mask'] = qformer_prompt['attention_mask'][0].astype(np.float32)

            # Add special token 
            intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            answers = random.choice(anno['answers'])
            answers = prompt['answer'].format(locations=boxes, answer=answers)
            llm_inputs = self.tokenizer.batch_encode_plus(
                [' '.join((intruction, answers, self.tokenizer.eos_token))],
                **self.tokenizer_config
            )
            
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            # ret_dict['start_learnable_id'] = np.array(np.where(ret_dict['gradient_mask']==1)[0][0]).astype(np.int64)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)

            return ret_dict
        
        # Scene Caption
        elif task_name == 'scene_understanding':
            intruction = anno['question']

            qformer_prompt = self.qtokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            ret_dict['qformer_instruction'] = qformer_prompt['input_ids'][0].astype(np.int64)
            ret_dict['qformer_instruction_mask'] = qformer_prompt['attention_mask'][0].astype(np.float32)

            intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            
            answers = anno['answers'][0]
            llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
            )
            
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
            # try:
            #     ret_dict['start_learnable_id'] = np.array(np.where(ret_dict['gradient_mask']==1)[0][0]).astype(np.int64)
            # except:
            #     ret_dict = self.last_scene_understanding_ret_dict
            #     print('Error: start_learnable_id ')
            # self.last_scene_understanding_ret_dict = ret_dict
            return ret_dict
        
        # Object Caption
        elif task_name == 'object_caption':
            instance_id = int(anno['target_id']) if 'target_id' in anno else int(anno['object_id'])
            
            if random.random() < 0.5 or self.config.subset == 'val':
                click_query[0] = random.choice(points[instance_labels == instance_id])
                click_mask[0] = 1
            else:
                box_mask[0] = 1
                box_query = features[instance_labels==instance_id].mean(0)
            ret_dict.update({
                'click_query': click_query.astype(np.float32),
                'click_mask': click_mask.astype(np.float32),
                'box_mask': box_mask.astype(np.float32),
                'box_query': box_query.astype(np.float32),
            })

            object_points = points[instance_labels==instance_id] 
            boxes = convert_objectpoints_to_bbox_str(points, object_points)
        
            if self.config.subset == 'train':
                prompt = deepcopy(random.choice(TASK_PROMPT[task_name]))
            else:
                prompt = deepcopy(TASK_PROMPT[task_name][0])
    
            intruction = prompt['instruction']

            qformer_prompt = self.qtokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            ret_dict['qformer_instruction'] = qformer_prompt['input_ids'][0].astype(np.int64)
            ret_dict['qformer_instruction_mask'] = qformer_prompt['attention_mask'][0].astype(np.float32)

            intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            caption = anno['utterance'] if 'utterance' in anno else anno['answers'][0]
            answers = prompt['answer'].format(locations=boxes, caption=caption)
            llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
            )
            
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            # ret_dict['start_learnable_id'] = np.array(np.where(ret_dict['gradient_mask']==1)[0][0]).astype(np.int64)
            return ret_dict
        
        elif task_name == 'hd_scene_qa':
        
            target_obj_id = int(anno['target_id'])
            object_points = points[instance_labels == target_obj_id]    # npt x 3
                
            if self.config.subset == 'train' and random.random() < 0.25 and len(object_points) > 0:
                click_query[0] = random.choice(object_points)
                click_mask[0] = 1
            ret_dict.update({
                'click_query': click_query.astype(np.float32),
                'click_mask': click_mask.astype(np.float32),
                'box_mask': box_mask.astype(np.float32),
                'box_query': box_query.astype(np.float32),
            })
            
            question = anno['question']
            
            if self.config.subset == 'train' and len(object_points) > 0:
                boxes = convert_objectpoints_to_bbox_str(points, object_points)
                prompt = deepcopy(random.choice(TASK_PROMPT[task_name])) if self.config.differ_prompt else deepcopy(random.choice(TASK_PROMPT['scene_qa']))
            else:
                prompt = deepcopy(TASK_PROMPT[task_name][0]) if self.config.differ_prompt else deepcopy(TASK_PROMPT['scene_qa'][0])
                boxes = ''
            
            intruction = prompt['instruction'].format(locations=boxes, question=question)
            
            qformer_prompt = self.qtokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            ret_dict['qformer_instruction'] = qformer_prompt['input_ids'][0].astype(np.int64)
            ret_dict['qformer_instruction_mask'] = qformer_prompt['attention_mask'][0].astype(np.float32)
            
            # Add special token 
            intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            answers = anno['answers'][0]
            answers = prompt['answer'].format(locations=boxes, answer=answers)
            llm_inputs = self.tokenizer.batch_encode_plus(
                [' '.join((intruction, answers, self.tokenizer.eos_token))],
                **self.tokenizer_config
            )
            
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
            # ret_dict['start_learnable_id'] = np.array(np.where(ret_dict['gradient_mask']==1)[0][0]).astype(np.int64)
            return ret_dict
        
        """ elif task_name == 'object_grouding' or task_name == 'object_caption_given_bbox':
            tgt_id = int(anno['target_id']) if 'target_id' in anno else int(anno['object_id'])
            
            if not self.OPENSCENE:
                points = _augment_pointcloud(points)
            
            # First check if target object's point less than 50 which will cause has no object point after downsample
            dense_object_points = points[instance_labels == tgt_id]
            dense_object_colors = colors[instance_labels == tgt_id]
            if len(dense_object_points) < 25:
                sample_points = self._npoint - len(dense_object_points)
                points, colors, instance_labels, features = self.down_sample(points, colors, instance_labels, features, npoint=sample_points)
                points = np.concatenate((dense_object_points, points), axis=0)
                colors = np.concatenate((dense_object_colors, colors), axis=0)
                points = self.pc_norm(points)
                object_points = dense_object_points
            else:
                points, colors, instance_labels, features = self.down_sample(points, colors, instance_labels, features, npoint=self._npoint)
                points = self.pc_norm(points)
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
            
            # points = np.concatenate([points, colors/255], 1)
            points = _padding_pointcloud(points)
            
            ret_dict = {
                'points': points.astype(np.float32),
                # 'colors': colors.astype(np.float32),
                'num_groups': self._num_groups,
                'group_size': self._group_size,
                'dataset_name': dataset_name,
                'level': 'scene',
                'scan_name': scan_name,
                'task_name': task_name,
                'episode_id': data['episode_id']
            }
            
            if self.OPENSCENE:
                ret_dict['features'] = _padding_pointcloud(features)
                ret_dict['valid_label'] = np.ones_like(instance_labels)
            
            caption = anno['utterance'] if 'utterance' in anno else anno['answers'][0]
            if task_name == 'object_grouding':
                intruction = random.choice(TASK_PROMPT[task_name]).format(caption=caption)
                answers = bbox_str
            else:
                intruction = random.choice(TASK_PROMPT[task_name]).format(bbox=bbox_str)
                answers = caption
                
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            
            llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
            )
            
            # ret_dict['answers'] = answers
            # ret_dict['question'] = intruction
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)

            return ret_dict """
            

@DATASETS.register_module()
class HD_Hm3dQADataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf', add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.qtokenizer = AutoTokenizer.from_pretrained('ckpts/bert-base-uncased')
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'
        
        self.config = config
        
        if not hasattr(self.config, 'differ_prompt'):
            self.config.differ_prompt = False
        print('Use Different Prompt:', self.config.differ_prompt)
        
        self.SCENE_TOKEN = '<scene><scene_placehold></scene>'
        self.VP_TOKEN = '<vp_placehold>'
        
        self.USE_OBJECTCENTRIC = config.get('USE_OBJECTCENTRIC', False)
        self.room_label_offset = 10000
        
        special_tokens = ['<vp_placehold>', '<scene>', '<scene_placehold>', '</scene>', '<obj>', '</obj>']
        # xyz_prompt = '<loc{}>'
        # for i in range(255):
        #     special_tokens.append(xyz_prompt.format(i))
        # whl_prompt = '<whl{}>'
        # for i in range(255):
        #     special_tokens.append(whl_prompt.format(i))
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
        
        # If use openscene as encoder
        self.OPENSCENE = config.OPENSCENE
        self._openscene_root = 'data/SceneVerse/OpenScene_Scan_Features'
        
        # Load scene level data
        self._npoint = config.N_POINTS
        self._group_size = config.GROUP_SIZE
        self._num_groups = config.NUM_GROUP
        self._all_dataset_root = 'data/SceneVerse'
        
        qa_source_dir_f = 'data/SceneVerse/HM3D/annotations/qa/qa_pairs/{}.json'.format(config.subset)
        # qa_source_dir_f = 'data/SceneVerse/HM3D/annotations/qa/qa_pairs/vis.json'
        with open(qa_source_dir_f, 'r') as f:
            datas = json.load(f)
        self.all_scene_qa = []
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
                    self.all_scene_qa.append({'dataset_name':'HM3D', 
                                            "scan_name":scan_name, 
                                            'instance_room_id': epi['scan_id'].split('_')[-1],
                                            "anno":anno, 
                                            "task_name": "hd_scene_qa",
                                            'region_id': qa['region_id'],
                                            'episode_id':'{}#{}#{}#{}'.format('HM3D', scan_name, qa['region_id'], qa['question'])
                                            })
        print_log(f'[DATASET] {len(self.all_scene_qa)} scene qa were loaded from HM3D scan data', logger = 'HD_Hm3dQADataset')
        
        # Prepare corpus for evaluation
        self.corpus = {
            'hd_scene_qa': copy.deepcopy(self.all_scene_qa),
        }

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
        if not self.OPENSCENE:
            dataset_root = os.path.join(self._all_dataset_root, dataset_name)
            annotation_root = os.path.join(dataset_root, 'annotations')
            scan_data_root = os.path.join(dataset_root, 'scan_data')
            
            inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
            pcd_path = os.path.join(scan_data_root,'pcd_with_global_alignment')

            points, colors, pcds, instance_labels, inst_to_label = self._load_scan(pcd_path, inst2label_path, scan_name)
            features = None
        else:
            dataset_root = os.path.join(self._all_dataset_root, dataset_name)
            scan_data_root = os.path.join(dataset_root, 'scan_data')
            inst2label_path = os.path.join(scan_data_root,'instance_id_to_label')
            inst_to_label = torch.load(os.path.join(inst2label_path, f"{scan_name}")) 
            dataset_root = os.path.join(self._openscene_root, dataset_name)
            dict = torch.load(os.path.join(dataset_root, scan_name), map_location='cpu')
            points = dict['points'].numpy().astype(np.float32)
            colors = dict['colors'].numpy()
            features = dict['features'].numpy().astype(np.float32)
            instance_labels = dict['instance_labels'].numpy()
        return points, colors, features, instance_labels, inst_to_label
    
    def __len__(self):
        return len(self.all_scene_qa)
    
    def __getitem__(self, index):
        
        data = self.all_scene_qa[index]
        dataset_name, scan_name, anno, task_name, region_id = data['dataset_name'], data['scan_name'], data['anno'], data['task_name'], data['region_id']
        instance_room_id = data['instance_room_id']
        tgt_id = int(anno['target_id'])
        room_center = torch.load(os.path.join('data/SceneVerse/HM3D/scan_data/room_center', '{}.pth'.format(scan_name)))
        
        self.tokenizer_config = dict(
            max_length=256, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        
        points = []
        colors = []
        features = []
        instance_labels = []
        inst_to_label = {}
        for ri, room_id in enumerate(region_id.split('-')):
            pts, cols, fts, ils, itl = self._load_scan_data(f'{scan_name}_{room_id}.pth', dataset_name)
            points.extend(pts + room_center[room_id]['center'])
            colors.extend(cols)
            features.extend(fts)
            if room_id == instance_room_id:
                instance_labels.extend(ils)
            else:
                if self.USE_OBJECTCENTRIC:
                    instance_labels.extend(ils+int(self.room_label_offset*ri))
                else:
                    instance_labels.extend(np.zeros_like(ils).astype(ils.dtype))
            inst_to_label[room_id] = itl
        
        points = np.array(points)
        points = pc_norm(points)
        colors = np.array(colors)
        features = np.array(features)
        instance_labels = np.array(instance_labels)


        # Get HD Info
        N = 160000
        hd_points, hd_features, hd_colors, hd_instance_labels = down_sample(points, features, colors, instance_labels, npoint=N)
        
        # Viusalization code  
        # print(inst_to_label[instance_room_id][tgt_id])
        # print(anno['utterance'])
        # visulization_pointclouds_bbox_use_plt(points, colors, points[instance_labels==tgt_id])
    
        points, colors, instance_labels, features = down_sample(points, colors, instance_labels, features, npoint=self._npoint)
        self.obj_pcd_num = 256 
        pcds = np.concatenate([points, colors/255], -1)
        obj_tokens, obj_center, obj_mask, obj_loc, obj_token_label = proc_object_tokens_for_objectcentric(pcds, instance_labels)

        click_query = np.zeros((1, 3))
        click_mask = np.zeros((1,))
        box_mask = np.zeros((1,))
        box_query = np.zeros((features.shape[-1],))
        ret_dict = {
            'num_groups': self._num_groups,
            'group_size': self._group_size,
            'dataset_name': dataset_name,
            'level': 'scene',
            'scan_name': scan_name,
            'task_name': task_name,
            'episode_id': data['episode_id'],
            'hd_points': hd_points.astype(np.float32),
            'hd_features': hd_features.astype(np.float32),
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
            'features': features.astype(np.float32),
            'click_query': click_query.astype(np.float32),
            'click_mask': click_mask.astype(np.float32),
            'box_mask': box_mask.astype(np.float32),
            'box_query': box_query.astype(np.float32),
            'hd_instance_labels': hd_instance_labels.astype(np.int64)
        }
        
        if self.USE_OBJECTCENTRIC:
            ret_dict['obj_tokens'] = obj_tokens.astype(np.float32)
            ret_dict['obj_center'] = obj_center.astype(np.float32)
            ret_dict['obj_mask'] = obj_mask.astype(np.int64)
            ret_dict['obj_loc'] = obj_loc.astype(np.float32)
            ret_dict['obj_token_label'] = np.array(obj_token_label).astype(np.int64)
        
        if hasattr(self.config, 'vis'):
            ret_dict['hd_colors'] = hd_colors.astype(np.float32)
            ret_dict['tgt_id'] = tgt_id
        
        question = anno['question']
        prompt = deepcopy(TASK_PROMPT[task_name][0]) if self.config.differ_prompt else deepcopy(TASK_PROMPT['scene_qa'][0])
        boxes = ''
        intruction = prompt['instruction'].format(locations=boxes, question=question)

        qformer_prompt = self.qtokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
        ret_dict['qformer_instruction'] = qformer_prompt['input_ids'][0].astype(np.int64)
        ret_dict['qformer_instruction_mask'] = qformer_prompt['attention_mask'][0].astype(np.float32)

        # Add special token 
        intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
        prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
        answers = anno['answers'][0]
        answers = prompt['answer'].format(locations=boxes, answer=answers)
        llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
        )
        
        ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
        ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['gradient_mask'] = \
            (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        # ret_dict['start_learnable_id'] = np.array(np.where(ret_dict['gradient_mask']==1)[0][0]).astype(np.int64)
        return ret_dict
    

@DATASETS.register_module()
class NuscenesDataset(Dataset):
    
    def gen_caption(self, reference_all_data):
        reference = reference_all_data['attribute_caption']['attribute_caption'] + \
                                " " + reference_all_data['localization_caption']['localization_caption'] + \
                                " is " + reference_all_data['motion_caption']['motion_caption'] + \
                                " " + reference_all_data['map_caption']['map_caption']
        return reference
    
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf', add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.config = config
        
        self.all_episodes = []
        
        self.SCENE_TOKEN = '<scene><scene_placehold></scene>'
        self.VP_TOKEN = '<vp_placehold>'
        special_tokens = ['<vp_placehold>', '<scene>', '<scene_placehold>', '</scene>', '<obj>', '</obj>']
        # xyz_prompt = '<loc{}>'
        # for i in range(255):
        #     special_tokens.append(xyz_prompt.format(i))
        # whl_prompt = '<whl{}>'
        # for i in range(255):
        #     special_tokens.append(whl_prompt.format(i))
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
        self._openscene_root = 'data/SceneVerse/OpenScene_Nuscenes_Features'
        all_fts_file = {item:[] for item in os.listdir(f"{self._openscene_root}/Nuscenes_Openscene")}
        # bad_token_list = ['3337be594faf4cfe895b9209bf54c91c', 'eb3a44c220cc4e9eb06bd69727da22de']
        
        if config.subset == 'train':
            if hasattr(self.config,"ALL_DATA"):
                num = 10000000
            else:
                num = 100000 if not hasattr(self.config, "QA_NUM") else self.config.QA_NUM 
        else:
            if hasattr(self.config,"ALL_DATA"):
                num = 10000000
            else:
                num = 4000 
        
        valid_sample_tokens = []
        qa_source_dir_f = 'data/NuscenesQA/questions/NuScenes_{}_questions.json'.format(config.subset)
        with open(qa_source_dir_f, 'r') as f:
            datas = json.load(f)['questions'][:num]
        self.all_qa = []
        for episodes in tqdm(datas):
            if '{}.pth'.format(episodes["sample_token"]) in all_fts_file.keys() or \
               '{}.npz'.format(episodes["sample_token"]) in all_fts_file.keys() :
                valid_sample_tokens.append(episodes["sample_token"])
                episodes['episode_id'] = '{}#{}#{}'.format(episodes["sample_token"], episodes["question"], episodes["answer"])
                episodes['anno'] = {'answers': [episodes['answer']]}
                episodes['task_name'] = 'nuscenes_qa'
                self.all_qa.append(episodes)
        valid_sample_tokens = set(valid_sample_tokens)
        self.all_episodes.extend(self.all_qa)
        if self.config.subset == 'val' and hasattr(os.environ, 'CHUNK'):
            CHUNK = int(os.environ['CHUNK'])
            self.all_episodes = self.all_episodes[int(len(self.all_episodes)//2)*CHUNK: int(len(self.all_episodes)//2)*(CHUNK+1)]
        print_log(f'[DATASET] {len(self.all_qa)} qa were loaded from NuscenesQA data', logger = 'NuscenesDataset')
        
        # Nuscenes Caption from TOD3Cap
        if hasattr(self.config, "USE_CAPTION_DATA") and self.config.subset == 'train':
            if not hasattr(self.config, "CAPTION_NUM"):
                filter_info = json.load(open('data/NuscenesCaption/filter_400k.json', 'r'))
            else:
                filter_info = json.load(open('data/NuscenesCaption/filter_400k.json', 'r'))[:self.config.CAPTION_NUM]
            instance_info  = json.load(open("data/NuscenesCaption/instance.json", "r"))
            instance_info = {ins['token']: ins['category_token'] for ins in instance_info}
            category_info  = json.load(open("data/NuscenesCaption/category.json", "r"))
            category_info = {cat['token']: cat['index'] for cat in category_info}
            
            captions = json.load(open("data/NuscenesCaption/final_caption_bbox_token.json", "r"))
            self.caption = []
            for _, cap in tqdm(captions.items()):
                if cap['sample_token'] in filter_info.keys():
                    if cap['sample_token'] in valid_sample_tokens and \
                        category_info[instance_info[cap['instance_token']]] in filter_info[cap['sample_token']]:
                        gt = self.gen_caption(cap)
                        self.caption.append(
                            {
                                'episode_id': '{}#{}'.format(cap["sample_token"], gt),
                                'answers': [gt],
                                'target_id': category_info[instance_info[cap['instance_token']]],
                                'anno': {'answers': [gt]},
                                'task_name': 'nuscenes_object_caption',
                                'sample_token': cap["sample_token"]
                            }  
                        )
            self.all_episodes.extend(self.caption)
            print_log(f'[DATASET] {len(self.caption)} caption were loaded from TOD3Cap data', logger = 'NuscenesDataset')

        self.corpus = {
            'nuscenes_qa': copy.deepcopy(self.all_qa),
        }

        self.tokenizer_config = dict(
            max_length=256, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        
    def __len__(self):
        return len(self.all_episodes)
    
    def __getitem__(self, index):
        
        data = self.all_episodes[index]
        task_name = data['task_name']
        
        if os.path.exists(os.path.join("data/SceneVerse/OpenScene_Nuscenes_Features/Nuscenes_Openscene", "{}.pth".format(data['sample_token']))):
            scene_dict = torch.load(os.path.join("data/SceneVerse/OpenScene_Nuscenes_Features/Nuscenes_Openscene", "{}.pth".format(data['sample_token'])), map_location='cpu')
            raw_points = scene_dict['points'].numpy().astype(np.float32)
            raw_colors = scene_dict['colors'].numpy()
            raw_features = scene_dict['features'].numpy().astype(np.float32)
            raw_instance_labels = scene_dict['instance_labels'].numpy()
        else:
            scene_dict = np.load(os.path.join("data/SceneVerse/OpenScene_Nuscenes_Features/Nuscenes_Openscene", "{}.npz".format(data['sample_token'])))
            raw_points = scene_dict['points'].astype(np.float32)
            raw_colors = np.zeros_like(raw_points)
            raw_features = scene_dict['features'].astype(np.float32)
            raw_instance_labels = scene_dict['instance_labels']
        raw_points = pc_norm(raw_points)
        
        hd_points, hd_features, hd_colors, hd_instance_labels = down_sample(raw_points, raw_features, raw_colors, raw_instance_labels, npoint=40000)
        points, colors, instance_labels, features = down_sample(raw_points, raw_colors, raw_instance_labels, raw_features, npoint=self.config.N_POINTS)
        
        click_query = np.zeros((1, 3))
        click_mask = np.zeros((1,))
        box_mask = np.zeros((1,))
        box_query = np.zeros((features.shape[-1],))
        ret_dict = {
            'num_groups': self.config.NUM_GROUP,
            'group_size': self.config.GROUP_SIZE,
            'level': 'scene',
            'episode_id': data['episode_id'],
            'hd_points': hd_points.astype(np.float32),
            'hd_features': hd_features.astype(np.float32),
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
            'features': features.astype(np.float32),
            'click_query': click_query.astype(np.float32),
            'click_mask': click_mask.astype(np.float32),
            'box_mask': box_mask.astype(np.float32),
            'box_query': box_query.astype(np.float32),
            'hd_instance_labels': hd_instance_labels.astype(np.int64),
            'task_name': task_name
        }
        
        if task_name == 'nuscenes_qa':
        
            question = data['question']
            prompt = deepcopy(TASK_PROMPT[task_name][0]) 
            boxes = ''
            intruction = prompt['instruction'].format(locations=boxes, question=question)

            # Add special token 
            intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            answers = data['answer']
            answers = prompt['answer'].format(locations=boxes, answer=answers)
            llm_inputs = self.tokenizer.batch_encode_plus(
                [' '.join((intruction, answers, self.tokenizer.eos_token))],
                **self.tokenizer_config
            )
            
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
            return ret_dict
        
        elif task_name == 'nuscenes_object_caption':
            instance_id = int(data['target_id'])

            if random.random() < 0.5 or self.config.subset == 'val':
                click_query[0] = random.choice(raw_points[raw_instance_labels == instance_id])
                click_mask[0] = 1
            else:
                box_mask[0] = 1
                box_query = raw_features[raw_instance_labels==instance_id].mean(0)
            ret_dict.update({
                'click_query': click_query.astype(np.float32),
                'click_mask': click_mask.astype(np.float32),
                'box_mask': box_mask.astype(np.float32),
                'box_query': box_query.astype(np.float32),
            })
            
            object_points = raw_points[raw_instance_labels==instance_id] 
            boxes = convert_objectpoints_to_bbox_str(raw_points, object_points)

            if self.config.subset == 'train':
                prompt = deepcopy(random.choice(TASK_PROMPT[task_name]))
            else:
                prompt = deepcopy(TASK_PROMPT[task_name][0])
    
            intruction = prompt['instruction']

            intruction = '{} {} {} {}'.format(SYSTEM_PROMPT, self.SCENE_TOKEN, self.VP_TOKEN, intruction)
            prompt_inputs = self.tokenizer.batch_encode_plus([intruction], **self.tokenizer_config)
            caption = data['answers'][0]
            answers = prompt['answer'].format(locations=boxes, caption=caption)
            llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((intruction, answers, self.tokenizer.eos_token))],
            **self.tokenizer_config
            )
            
            ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
            ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            return ret_dict