import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


@DATASETS.register_module()
class Objaverse(Dataset):

    def __init__(self, num_points, split):
        super().__init__()
        self.split = split
        self._npoint = num_points
        self.create_obj_cap_dict('data/LEO_DATA/Cap3d_root/Cap3D_pcs_pt')
        # if split == 'train':
        #     self.obj_ids = self.obj_ids[:-1000]
        # else:
        #     self.obj_ids = self.obj_ids[-1000:]
        logger.info(f"Finish loading Objaverse {split}-set language, collected {len(self.obj_ids)} data")

    def create_obj_cap_dict(self, cap3d_root):
        obj_csv = pd.read_csv(os.path.join(cap3d_root, 'Cap3D_automated_Objaverse_no3Dword.csv'), header=None)
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
        pcd[:, 3:] = pcd[:, 3:] / 127.5 - 1   # (16384, 6), xyz (m) + rgb (float, [-1, 1])
        pcd = self.down_sample(pcd)
        pcd[:, :3] = self.pc_norm(pcd[:, :3])
        return pcd

    def down_sample(self, pcd)
        pcd_idxs = np.random.choice(len(pcd), size=self._npoint, replace=len(pcd) < self._npoint)
        pcd = pcd[pcd_idxs]
        return pcd
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, index):
        obj_id = self.obj_ids[index]
        obj_pcd = self.load_obj_pcd(obj_id)

        # TODO
        

class SceneVerseDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        self._npoint = 4e4
        self._dataset_name = dataset_name
        self._all_dataset_root = 'data/SceneVerse'
        self.all_scans = (self._all_dataset_root / 'scan_data' / 'pcd_with_global_alignment').glob('*.pth')
    
    def _load_scan(pcd_path, inst2label_path, scan_name):
        pcd_data = torch.load(pcd_path / f'{scan_name}.pth')
        inst_to_label = torch.load(inst2label_path / f"{scan_name}.pth")
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        
        points, colors, instance_labels = self.down_sample(points, colors, instance_labels)
        points = self.pc_norm(points)
        
        pcds = np.concatenate([points, colors], 1)
        return points, colors, pcds, instance_labels, inst_to_label
    
    def _load_scan_data(self, scan_name):
        dataset_root = os.path.join(self._all_dataset_root, self._dataset_name)
        annotation_root = os.path.join(dataset_root, 'annotations')
        scan_data_root = os.path.join(dataset_root, 'scan_data')
        
        inst2label_path = scan_data_root / 'instance_id_to_label'
        pcd_path = scan_data_root / 'pcd_with_global_alignment'

        points, colors, pcds, instance_labels, inst_to_label = self._load_scan(pcd_path, inst2label_path, scan_name)
        
    def convert_pc_to_box(obj_pc):
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
        box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
        return center, box_size
    
    def down_sample(self, points, colors, instance_labels)
        pcd_idxs = np.random.choice(len(points), size=self._npoint, replace=len(points) < self._npoint)
        points = points[pcd_idxs]
        colors = colors[pcd_idxs]
        instance_labels = instance_labels[pcd_idxs]
        return points, colors, instance_labels
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def __len__(self):
        return len(self.all_scans)
        
    def __getitem__(self, index):
        scan_name = self.all_scans[index]
        points, colors, pcds, instance_labels, inst_to_label = self._load_scan_data(scan_name)
        # TODO
        
        
class RScanDataset(SceneVerseDataset):
    def __init__(self):
        super().__init__('3RScan')
        

class HM3DDataset(SceneVerseDataset):
    def __init__(self):
        super().__init__('HM3D')
        
    
class MultiScanDataset(SceneVerseDataset):
    def __init__(self):
        super().__init__('MultiScan')
        

class ARKitScenesDataset(SceneVerseDataset):
    def __init__(self):
        super().__init__('ARKitScenes')


# TODO: Region augment dataset