# import sys
# sys.path.append('/gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/Adaptive_3D_LLM/')
# import utils.capeval.bleu.bleu as capblue
# import utils.capeval.cider.cider as capcider
# import utils.capeval.rouge.rouge as caprouge
# import utils.capeval.meteor.meteor as capmeteor
# import torch
# from collections import defaultdict, OrderedDict
# import os
# import json

# def score_captions(corpus: dict, candidates: dict):
    
#     bleu = capblue.Bleu(4).compute_score(corpus, candidates)
#     cider = capcider.Cider().compute_score(corpus, candidates)
#     rouge = caprouge.Rouge().compute_score(corpus, candidates)
#     meteor = capmeteor.Meteor().compute_score(corpus, candidates)
    
#     score_per_caption = {
#         "bleu-1": [float(s) for s in bleu[1][0]],
#         "bleu-2": [float(s) for s in bleu[1][1]],
#         "bleu-3": [float(s) for s in bleu[1][2]],
#         "bleu-4": [float(s) for s in bleu[1][3]],
#         "cider": [float(s) for s in cider[1]],
#         "rouge": [float(s) for s in rouge[1]],
#         "meteor": [float(s) for s in meteor[1]],
#     }
    
#     message = '\n'.join([
#         "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
#             bleu[0][0], max(bleu[1][0]), min(bleu[1][0])
#         ),
#         "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
#             bleu[0][1], max(bleu[1][1]), min(bleu[1][1])
#         ),
#         "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
#             bleu[0][2], max(bleu[1][2]), min(bleu[1][2])
#         ),
#         "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
#             bleu[0][3], max(bleu[1][3]), min(bleu[1][3])
#         ),
#         "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
#             cider[0], max(cider[1]), min(cider[1])
#         ),
#         "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
#             rouge[0], max(rouge[1]), min(rouge[1])
#         ),
#         "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
#             meteor[0], max(meteor[1]), min(meteor[1])
#         )
#     ])
    
#     eval_metric = {
#         "BLEU-4": bleu[0][3],
#         "CiDEr": cider[0],
#         "Rouge": rouge[0],
#         "METEOR": meteor[0],
#     }
#     return score_per_caption, message, eval_metric

# exp_path = '/gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/pointbert/experiments/Adaptive-LLM/MultiScale_models/multiscale_llm_pretrain_accgrad_validation_0.pth'

# exp = torch.load(exp_path)
# scanqa_corpus = exp['scanqa_corpus']
# object_caption_corpus = exp['object_caption_corpus']
# scene_caption_corpus = exp['scene_caption_corpus']

# candidates = exp['candidates']
# # print(list(candidates['scanqa'].values())[:10])
# # print(list(candidates['scene_caption'].values())[:10])
# qa_score_per_caption, qa_message, qa_eval_metric = score_captions(
#             OrderedDict([(key, scanqa_corpus[key]) for key in candidates['scanqa']]), candidates['scanqa']
#         )
# # oc_score_per_caption, oc_message, oc_eval_metric = score_captions(
# #     OrderedDict([(key, object_caption_corpus[key]) for key in candidates['object_caption']]), candidates['object_caption']
# # )
# sc_score_per_caption, sc_message, sc_eval_metric = score_captions(
#     OrderedDict([(key, scene_caption_corpus[key]) for key in candidates['scene_caption']]), candidates['scene_caption']
# )


# print("\n----------------------Evaluation QA-----------------------\n")
# print(qa_message)
# # print_log("\n----------------------Evaluation OC-----------------------\n")
# # print_log(oc_message)
# print("\n----------------------Evaluation SC-----------------------\n")
# print(sc_message)


# # with open(os.path.join('/gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/Adaptive_3D_LLM/results', f"Scanqa_qa_pred_gt_val.json"), "w") as f:
# #     pred_gt_val = {}
# #     qa_candidates = candidates['scanqa']
# #     # print(qa_candidates)
# #     for scene_object_id, scene_object_id_key in enumerate(qa_candidates):
# #         pred_gt_val[scene_object_id_key] = {
# #             'pred': qa_candidates[scene_object_id_key],
# #             'gt': scanqa_corpus[scene_object_id_key],
# #             'score': {
# #                 'bleu-1': qa_score_per_caption['bleu-1'][scene_object_id],
# #                 'bleu-2': qa_score_per_caption['bleu-2'][scene_object_id],
# #                 'bleu-3': qa_score_per_caption['bleu-3'][scene_object_id],
# #                 'bleu-4': qa_score_per_caption['bleu-4'][scene_object_id],
# #                 'CiDEr': qa_score_per_caption['cider'][scene_object_id],
# #                 'rouge': qa_score_per_caption['rouge'][scene_object_id],
# #                 'meteor': qa_score_per_caption['meteor'][scene_object_id]
# #             }
# #         }
# #     json.dump(pred_gt_val, f, indent=4)
    
    
    
# # with open(os.path.join('/gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/Adaptive_3D_LLM/results', f"SceneCaption_qa_pred_gt_val.json"), "w") as f:
# #     pred_gt_val = {}
# #     qa_candidates = candidates['scene_caption']
# #     # print(qa_candidates)
# #     for scene_object_id, scene_object_id_key in enumerate(qa_candidates):
# #         pred_gt_val[scene_object_id_key] = {
# #             'pred': qa_candidates[scene_object_id_key],
# #             'gt': scene_caption_corpus['-'.join((scene_object_id_key))],
# #             'score': {
# #                 'bleu-1': sc_score_per_caption['bleu-1'][scene_object_id],
# #                 'bleu-2': sc_score_per_caption['bleu-2'][scene_object_id],
# #                 'bleu-3': sc_score_per_caption['bleu-3'][scene_object_id],
# #                 'bleu-4': sc_score_per_caption['bleu-4'][scene_object_id],
# #                 'CiDEr': sc_score_per_caption['cider'][scene_object_id],
# #                 'rouge': sc_score_per_caption['rouge'][scene_object_id],
# #                 'meteor': sc_score_per_caption['meteor'][scene_object_id]
# #             }
# #         }
# #     json.dump(pred_gt_val, f, indent=4)
        
        
import sys
sys.path.append('/gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/pointbert/')
import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor
import torch
from collections import defaultdict, OrderedDict

def score_captions(corpus: dict, candidates: dict):
    
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)
    
    score_per_caption = {
        "bleu-1": [float(s) for s in bleu[1][0]],
        "bleu-2": [float(s) for s in bleu[1][1]],
        "bleu-3": [float(s) for s in bleu[1][2]],
        "bleu-4": [float(s) for s in bleu[1][3]],
        "cider": [float(s) for s in cider[1]],
        "rouge": [float(s) for s in rouge[1]],
        "meteor": [float(s) for s in meteor[1]],
    }
    
    message = '\n'.join([
        "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][0], max(bleu[1][0]), min(bleu[1][0])
        ),
        "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][1], max(bleu[1][1]), min(bleu[1][1])
        ),
        "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][2], max(bleu[1][2]), min(bleu[1][2])
        ),
        "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][3], max(bleu[1][3]), min(bleu[1][3])
        ),
        "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            cider[0], max(cider[1]), min(cider[1])
        ),
        "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            rouge[0], max(rouge[1]), min(rouge[1])
        ),
        "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            meteor[0], max(meteor[1]), min(meteor[1])
        )
    ])
    
    eval_metric = {
        "BLEU-4": bleu[0][3],
        "CiDEr": cider[0],
        "Rouge": rouge[0],
        "METEOR": meteor[0],
    }
    return score_per_caption, message, eval_metric

exp_path = 'experiments/Adaptive-LLM/MultiScale_models/multiscale_llm_pretrain_accgrad_validation_1.pth'

exp = torch.load(exp_path)
scanqa_corpus = exp['scanqa_corpus']
object_caption_corpus = exp['object_caption_corpus']
scene_caption_corpus = exp['scene_caption_corpus']
candidates = exp['candidates']

qa_score_per_caption, qa_message, qa_eval_metric = score_captions(
            OrderedDict([(key, scanqa_corpus[key]) for key in candidates['scanqa']]), candidates['scanqa']
        )
# oc_score_per_caption, oc_message, oc_eval_metric = score_captions(
#     OrderedDict([(key, object_caption_corpus[key]) for key in candidates['object_caption']]), candidates['object_caption']
# )
sc_score_per_caption, sc_message, sc_eval_metric = score_captions(
    OrderedDict([(key, scene_caption_corpus[key]) for key in candidates['scene_caption']]), candidates['scene_caption']
)


print("\n----------------------Evaluation QA-----------------------\n")
print(qa_message)
# print_log("\n----------------------Evaluation OC-----------------------\n")
# print_log(oc_message)
print("\n----------------------Evaluation SC-----------------------\n")
print(sc_message)