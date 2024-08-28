# '''Dataloader for fused point features.'''

import copy
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA
import os

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
        self.aug = False
        self.input_color = True # decide whether we use point color values as input

        # token_timestamp_map = json.load(open('nuscenes-full/v1.0-trainval/sample.json'))
        # token2timestamp = {}
        # for tt in token_timestamp_map:
        #     token2timestamp[str(tt['token'])] = tt['timestamp']
        import json
        corpus = []
        corpus.extend(json.load(open("../../data/NuscenesQA/questions/NuScenes_train_questions.json","r"))["questions"])
        # corpus.extend(json.load(open("../../data/NuscenesQA/questions/NuScenes_val_questions.json","r"))["questions"][4000:])
        all_fts_file = {item:[] for item in os.listdir("../../data/SceneVerse/OpenScene_Nuscenes_Features/Nuscenes_Openscene")}
        all_corpus_tokens ={}
        for q in corpus:
            if not "{}.pth".format(q["sample_token"]) in all_fts_file.keys() and \
                not "{}.npz".format(q["sample_token"]) in all_fts_file.keys():
                all_corpus_tokens[q["sample_token"]] = []
        
        # val_pcd = glob("/gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/pointbert/models/openscene/Nuscenes_Openscene/nuscenes_3d/val/*.pth")
        train_pcd = glob("/gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/pointbert/models/openscene/Nuscenes_Openscene/nuscenes_3d/train/*.pth")
        all_tokens = []
        all_tokens.extend(train_pcd)
        # all_tokens.extend(val_pcd)
        
        self.all_tokens = [tok for tok in all_tokens if tok.split('/')[-1].split('.')[0] in all_corpus_tokens.keys()]
        
        print('Total number of data:', len(self.all_tokens))  

    def __len__(self):
        return len(self.all_tokens)

    def __getitem__(self, index_long):

        index = index_long 
        try:
            data = torch.load(self.all_tokens[index])
        except:
            print(self.all_tokens[index])
            index = index-10
            data = torch.load(self.all_tokens[index])
        locs_in, feats_in, labels_in = data[0], data[1], data[-1] # xyz, rgb, instance_label
        
        labels_in[labels_in == -100] = 255
        labels_in = labels_in.astype(np.uint8)
        # no color in the input point cloud, e.g nuscenes
        if np.isscalar(feats_in) and feats_in == 0:
            feats_in = np.zeros_like(locs_in)
        feats_in = (feats_in + 1.) * 127.5
        
        # labels_in[labels_in < 0] = 0
        # if labels_in is None:
        #     labels_in = np.zeros_like(locs_in[:, 0])

        locs = locs_in
        mask_chunk = torch.ones_like(torch.from_numpy(labels_in)).bool()
        mask = torch.ones_like(torch.from_numpy(labels_in)).bool()
        self.voxelizer.use_augmentation = False
        locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
            locs[mask_chunk], feats_in[mask_chunk], labels_in[mask_chunk], return_ind=True)
        vox_ind = torch.from_numpy(vox_ind)
        mask = mask[vox_ind]

        labels = labels_in
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()
        
        return coords, feats, labels, None, mask, torch.from_numpy(inds_reconstruct).long(), torch.from_numpy(locs_in), torch.from_numpy(feats_in), torch.from_numpy(labels_in).long(), self.all_tokens[index]

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
    coords, feats, labels, feat_3d, mask, inds_reconstruct, locs_in, feat_in, labels_in, scene_name = list(zip(*batch))


    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        feat_3d, torch.cat(mask), torch.cat(inds_reconstruct), torch.cat(locs_in), torch.cat(feat_in), torch.cat(labels_in), scene_name