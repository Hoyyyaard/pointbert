import argparse
import json
import os

# ckpt into argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_gt_json', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True, choices=['scanqa', 'hm3dqa','hm3dqa1'])
args = parser.parse_args()

map = {
    'scanqa': 'data/SceneVerse/meta_data/ScanQA_Bbox_Small_Than-0.05_Val.json',
    'hm3dqa': 'data/SceneVerse/meta_data/HDHm3dQA_Bbox_Small_Than-0.01_Val.json',
    'hm3dqa1': 'data/SceneVerse/meta_data/HDHm3dQA_Bbox_Small_Than-0.05_Val.json'
}

small_bbox_epi_ids = json.load(open(map[args.dataset]))
pred_gt = json.load(open(args.pred_gt_json))

small_scores = {
    "bleu-1": [],
    "bleu-2": [],
    "bleu-3": [],
    "bleu-4": [],
    "CiDEr": [],
    "rouge": [],
    "meteor": []
}

other_scores = {
    "bleu-1": [],
    "bleu-2": [],
    "bleu-3": [],
    "bleu-4": [],
    "CiDEr": [],
    "rouge": [],
    "meteor": []
}

all_scores = {
    "bleu-1": [],
    "bleu-2": [],
    "bleu-3": [],
    "bleu-4": [],
    "CiDEr": [],
    "rouge": [],
    "meteor": []
}

for k,epi in pred_gt.items():
    for key in all_scores.keys():
        all_scores[key].append(epi['score'][key])
    if k in small_bbox_epi_ids:
        for key in small_scores.keys():
            small_scores[key].append(epi['score'][key])
    else:
        for key in other_scores.keys():
            other_scores[key].append(epi['score'][key]) 
print('small_bbox_epi_ids:', len(small_bbox_epi_ids))
print("###################### Small Scores ######################")
print('small_scores:', {k: sum(v) / len(v) for k, v in small_scores.items()})
print("###################### Other Scores ######################")
print('other_scores:', {k: sum(v) / len(v) for k, v in other_scores.items()})
print("###################### All Scores ######################")
print('all_scores:', {k: sum(v) / len(v) for k, v in all_scores.items()})





