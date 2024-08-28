import json
import os


base_path = 'experiments/Adaptive-LLM-finetune-Openscene-test-FLEX-threshold/nuscenes/test_Exp0176_0823_Pts10k_400k_Token64_FlexWarmUp20_FlexThreshold127_From[Scratch]_Epoch5'

chunk0 = json.load(open(f'{base_path}_CHUNK0/nuscenes_qa_-1_qa_pred_gt_val.json'))
chunk1 = json.load(open(f'{base_path}_CHUNK1/nuscenes_qa_-1_qa_pred_gt_val.json'))

all_results = {}
all_results.update(chunk0)
all_results.update(chunk1)
os.makedirs(f'{base_path}_ALL', exist_ok=True)
with open(f'{base_path}_ALL/nuscenes_qa_-1_qa_pred_gt_val.json', 'w') as f:
    json.dump(all_results, f)