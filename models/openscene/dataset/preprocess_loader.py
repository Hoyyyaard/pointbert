'''Dataloader for fused point features.'''

import copy
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA
import open3d

from dataset.point_loader import Point3DLoader

class FusedFeatureLoader(Point3DLoader):
    '''Dataloader for fused point features.'''

    def __init__(self,
                 datapath_prefix,
                 datapath_prefix_feat,
                 voxel_size=0.05,
                 split='train', aug=False, memcache_init=False,
                 identifier=7791, loop=1, eval_all=False,
                 input_color = False,
                 ):
        super().__init__(datapath_prefix=datapath_prefix, voxel_size=voxel_size,
                                           split=split, aug=aug, memcache_init=memcache_init,
                                           identifier=identifier, loop=loop,
                                           eval_all=eval_all, input_color=input_color)
        self.aug = aug
        self.input_color = input_color # decide whether we use point color values as input

        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        # Precompute the occurances for each scene
        # for training sets, ScanNet and Matterport has 5 each, nuscene 1
        # for evaluation/test sets, all has just one
        if 'nuscenes' in self.dataset_name: # only one file for each scene
            self.list_occur = None
        else:
            self.list_occur = []
            for data_path in self.data_paths:
                if 'scannet' in self.dataset_name:
                    scene_name = data_path[:-15].split('/')[-1]
                else:
                    scene_name = data_path[:-4].split('/')[-1]
                    scene_name = data_path[:-4].split('/')[-1]
                file_dirs = glob(join(self.datapath_feat, scene_name + '_*.pt'))
                self.list_occur.append(len(file_dirs))
            # some scenes in matterport have no features at all
            ind = np.where(np.array(self.list_occur) != 0)[0]
            if np.any(np.array(self.list_occur)==0):
                data_paths, list_occur = [], []
                for i in ind:
                    data_paths.append(self.data_paths[i])
                    list_occur.append(self.list_occur[i])
                self.data_paths = data_paths
                self.list_occur = list_occur

        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the feature loader.')

        self.data_paths = [d for d in self.data_paths if d.split('/')[-1].split('_')[1] == '00']
        pass

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index_long):

        index = index_long 
        locs_in, feats_in, labels_in = torch.load(self.data_paths[index]) # xyz, rgb, instance_label
        
        ## openscene 的坐标没有转换到scannet
        scan_name = self.data_paths[index][:-15].split('/')[-1]
        # import os
        # opcd = np.load(os.path.join('/home/admin/Projects/LL3DA/data/scannet/scannet_data_w_sm_obj_dense', scan_name + '_ins_label.npy'))
        # assert opcd.shape[0] == locs_in.shape[0]
        
        meta_file = f'/mnt/nfs/share/datasets/scannet/scans/{scan_name}/{scan_name}.txt'
        import os
        if os.path.exists(meta_file):
            lines = open(meta_file).readlines()
            axis_align_matrix = None
            for line in lines:
                if 'axisAlignment' in line:
                    axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
            pts = np.ones((locs_in.shape[0], 4))
            pts[:,0:3] = locs_in[:,0:3]
            locs_in = np.dot(pts, axis_align_matrix.transpose())[:, 0:3] # Nx4
        else:
            print(scan_name)
            
        labels_in[labels_in == -100] = 255
        labels_in = labels_in.astype(np.uint8)
        if np.isscalar(feats_in) and feats_in == 0:
            # no color in the input point cloud, e.g nuscenes lidar
            feats_in = np.zeros_like(locs_in)
        else:
            feats_in = (feats_in + 1.) * 127.5

        # load 3D features
        if self.dataset_name == 'scannet_3d':
            scene_name = self.data_paths[index][:-15].split('/')[-1]
        else:
            scene_name = self.data_paths[index][:-4].split('/')[-1]

        if 'nuscenes' not in self.dataset_name:
            n_occur = self.list_occur[index]
            if n_occur > 1:
                nn_occur = np.random.randint(n_occur)
            elif n_occur == 1:
                nn_occur = 0
            else:
                raise NotImplementedError

            processed_data = torch.load(join(
                self.datapath_feat, scene_name+'_%d.pt'%(nn_occur)))
        else:
            # no repeated file
            processed_data = torch.load(join(self.datapath_feat, scene_name+'.pt'))

        flag_mask_merge = False
        if len(processed_data.keys())==2:
            flag_mask_merge = True
            feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
            ## sm code
            pad_num = locs_in.shape[0] - len(mask_chunk)
            if pad_num > 0:
                mask_chunk = np.concatenate((mask_chunk, np.zeros(pad_num, dtype=np.bool)))
            if isinstance(mask_chunk, np.ndarray): # if the mask itself is a numpy array
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            feat_3d_new = torch.zeros((locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype)
            feat_3d_new[mask] = feat_3d
            feat_3d = feat_3d_new
            mask_chunk = torch.ones_like(mask_chunk) # every point needs to be evaluted
        elif len(processed_data.keys())>2: # legacy, for old processed features
            feat_3d, mask_visible, mask_chunk = processed_data['feat'], processed_data['mask'], processed_data['mask_full']
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = True # mask out points without feature assigned

        if len(feat_3d.shape)>2:
            feat_3d = feat_3d[..., 0]

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in

        
        locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
            locs[mask_chunk], feats_in[mask_chunk], labels_in[mask_chunk], return_ind=True)
        vox_ind = torch.from_numpy(vox_ind)
        feat_3d = feat_3d[vox_ind]
        mask = mask[vox_ind]

        if self.eval_all: # during evaluation, no voxelization for GT labels
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()
        
        return coords, feats, labels, feat_3d, mask, torch.from_numpy(inds_reconstruct).long(), torch.from_numpy(locs_in), scene_name

def collation_fn(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    '''
    coords, feats, labels, feat_3d, inds_reconstruct, mask_chunk = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask_chunk)


def collation_fn_eval_all(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    '''
    coords, feats, labels, feat_3d, mask, inds_reconstruct, locs_in, scene_name = list(zip(*batch))


    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask), torch.cat(inds_reconstruct), torch.cat(locs_in), scene_name