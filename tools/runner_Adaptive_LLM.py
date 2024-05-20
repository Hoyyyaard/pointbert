import torch
import torch.nn as nn
import os
import json
import datetime
from tools import builder
from utils import misc, dist_utils
import time
from tqdm import tqdm
from utils.logger import *
from utils.AverageMeter import AverageMeter
from collections import defaultdict, OrderedDict
from utils.metrics import Metrics
from models.adaptive_llm import AdaptiveLLM
from transformers import AutoConfig
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from tools.dist import all_gather_dict, barrier, is_primary

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)

    # build model
    llm_config  = AutoConfig.from_pretrained('ckpts/Llama-2-7b-hf/config.json')
    base_model = AdaptiveLLM(llm_config, config.model)

    # from IPython import embed; embed()
    
    if args.use_gpu:
        base_model.to(args.local_rank)
    # print(torch.cuda.memory_summary())
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts, args)
        else:
            print_log('Training from scratch', logger = logger)
    # print(torch.cuda.memory_summary())

    # DDP
    # Debug
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, 
                                                        device_ids=[args.local_rank % torch.cuda.device_count()], 
                                                        find_unused_parameters=True,)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    # print(torch.cuda.memory_summary())
    
    # optimizer & scheduler
    finetune = config.get('finetune', False)
    optimizer, scheduler, warmup_steps = builder.build_llm_opti_sche(base_model, config, train_dataloader, finetune)
    print_log("Trainable parameters")
    for n, p in base_model.module.named_parameters():
        if p.requires_grad:
            print_log(n)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # Update learning rate at each step
    opti_config = config.optimizer
    sche_config = config.scheduler
    total_epochs = sche_config.kwargs.epochs
    warmup_ratio = opti_config.kwargs.warmup_ratio  # Warmup ratio of 3%
    initial_lr = opti_config.kwargs.lr  # Initial learning rate
    warmup_lr = 1e-6

    # Define scheduler with warmup
    total_steps = len(train_dataloader) * total_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    # 启用自动微分异常检测
    # torch.autograd.set_detect_anomaly(True)
    
    # trainval
    # training
    epoch_tqdm = tqdm(total = config.max_epoch, desc = 'Epoch', position = 0)
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        epoch_tqdm.update(1)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])
        from utils.misc import SmoothedValue
        time_delta = SmoothedValue(window_size=10)

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        pbar = tqdm(total = n_batches)
        for idx, data_dict in enumerate(train_dataloader):
            
            pbar.update(1)
        
            curr_time = time.time()
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            for k,v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.cuda()
            
            # Debug
            # break

            data_time.update(time.time() - batch_start_time)

            # assert points.size(1) == npoints
            # As we have some grouding episode which do not suitable for pointcloud aug
            # points = train_transforms(points)
            
            loss = base_model(data_dict)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=base_model.module.parameters(), max_norm=10, norm_type=2)
            
            # for name, param in base_model.module.named_parameters():
            #     if torch.isnan(param).any():
            #         pass
            
            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
                
            # for name, param in base_model.module.named_parameters():
            #     if torch.isnan(param).any():
            #         pass

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()])
            else:
                losses.update([loss.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            time_delta.update(time.time() - curr_time)

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
                eta_seconds = (n_batches - idx) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print_log(f'ETA: {eta_str}', logger = logger)
            
                
            # Define lambda function for warmup scheduler
            lr_step = epoch * len(train_dataloader) + idx
            if lr_step < warmup_steps:
                optimizer.param_groups[0]['lr'] =  warmup_lr 
            else:
                optimizer.param_groups[0]['lr'] =  0.5 * (math.cos((lr_step - warmup_steps) / (total_steps - warmup_steps) * math.pi) + 1) * initial_lr
        
        
                
        # Debug
        # if isinstance(scheduler, list):
        #     for item in scheduler:
        #         item.step(epoch)
        # else:
        #     scheduler.step(epoch)
        
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        # Debug
        # metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
        builder.save_checkpoint_pretrain_llm(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)  
        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint_pretrain_llm(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger) 
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint_pretrain_llm(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


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


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, finetune=False):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    
    if not finetune:
        candidates = {}
        test_pbar = tqdm(total = len(test_dataloader))
        with torch.no_grad():
            for idx, data_dict in enumerate(test_dataloader):
                test_pbar.update(1)
                
                for k,v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        data_dict[k] = v.cuda()
                
                output_ids = base_model(data_dict, eval=True)
                
                outputs = dict(
                    output_ids=output_ids,
                )
                
                outputs = all_gather_dict(outputs)
                data_dict = all_gather_dict(data_dict)
                # Flatten 2d list to 1d
                for k,v in data_dict.items():
                    if isinstance(v, list):
                        data_dict[k] = [item for sublist in zip(*v) for item in sublist]
                
                output_ids = outputs["output_ids"]  # batch x max_length
                answers = base_model.module.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                for idx in range(output_ids.shape[0]):
                    key = data_dict['episode_id'][idx].item()
                    task_name = data_dict['task_name'][idx]
                    answer = answers[idx]
                    answer = ' '.join(filter(lambda w: w, answer.split(' ')))
                    if not task_name in candidates:
                        candidates[task_name] = {key: [answer]}
                    else:
                        candidates[task_name][key] = [answer]
                
                barrier()
                # break
                
            for k, v in candidates.items():
                corpus = test_dataloader.dataset.corpus[k]
                new_corpus = {}
                for cor in corpus:
                    new_corpus[cor['episode_id']] = cor['anno']['utterance'] if 'utterance' in cor['anno'].keys() else cor['anno']['answers']
                corpus = new_corpus
                score_per_caption, message, eval_metric = score_captions(
                    OrderedDict([(key, corpus[key]) for key in v]), v
                )
            
                if is_primary():
                    print_log(f"\n----------------------Evaluation {k}-----------------------\n")
                    print_log(message)
                    _log_to_disk(args, message, corpus, v, score_per_caption, k, epoch)
                
            if args.distributed:
                torch.cuda.synchronize()
    
    '''            
    else:
        # Prepare ground truth for eval
        scanqa_anno = test_dataloader.dataset.scanqa_anno
        scanqa_corpus = {
            '-'.join((anno['question_id'], anno['question'])): anno['answers'] \
                for anno in scanqa_anno
        }
        object_caption_anno = test_dataloader.dataset.object_caption_anno
        object_caption_corpus = {
            '-'.join((anno['scene_id'], anno['object_id'])): anno['answers'] \
                for anno in object_caption_anno
        }
        scene_caption_anno = test_dataloader.dataset.scene_caption_anno
        scene_caption_corpus = {
            anno['scene_id']: anno['answers'] \
                for anno in scene_caption_anno
        }
        
        candidates = {
            'scanqa': {},
            'object_caption': {},
            'scene_caption': {}
        }
        
        test_pbar = tqdm(total = len(test_dataloader))
        with torch.no_grad():
            for idx, data_dict in enumerate(test_dataloader):
                test_pbar.update(1)
                
                for k,v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        data_dict[k] = v.cuda()
                
                output_ids = base_model(data_dict, eval=True)
                
                outputs = dict(
                    output_ids=output_ids,
                )
                
                outputs = all_gather_dict(outputs)
                data_dict = all_gather_dict(data_dict)
                # Flatten 2d list to 1d
                for k,v in data_dict.items():
                    if isinstance(v, list):
                        data_dict[k] = [item for sublist in zip(*v) for item in sublist]
                
                output_ids = outputs["output_ids"]  # batch x max_length
                answers = base_model.module.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                
                for idx in range(output_ids.shape[0]):
                    key = data_dict['unique_key'][idx]
                    task_name = data_dict['task_name'][idx]
                    answer = answers[idx]
                    answer = ' '.join(filter(lambda w: w, answer.split(' ')))
                    candidates[task_name][key] = [answer]
                
                barrier()

            torch.save({'candidates':candidates,
                        'scanqa_corpus':scanqa_corpus,
                        'object_caption_corpus':object_caption_corpus,
                        'scene_caption_corpus':scene_caption_corpus}, f'{args.experiment_path}_{args.exp_name}_validation_{epoch}.pth')
            print_log("Save all validation datas")

            # end of forward pass traversion
            qa_score_per_caption, qa_message, qa_eval_metric = score_captions(
                OrderedDict([(key, scanqa_corpus[key]) for key in candidates['scanqa']]), candidates['scanqa']
            )
            oc_score_per_caption, oc_message, oc_eval_metric = score_captions(
                OrderedDict([(key, object_caption_corpus[key]) for key in candidates['object_caption']]), candidates['object_caption']
            )
            sc_score_per_caption, sc_message, sc_eval_metric = score_captions(
                OrderedDict([(key, scene_caption_corpus[key]) for key in candidates['scene_caption']]), candidates['scene_caption']
            )
            
            if is_primary():
                print_log("\n----------------------Evaluation QA-----------------------\n")
                print_log(qa_message)
                # print_log("\n----------------------Evaluation OC-----------------------\n")
                # print_log(oc_message)
                print_log("\n----------------------Evaluation SC-----------------------\n")
                print_log(sc_message)

                _log_to_disk(args, qa_message, scanqa_corpus, candidates['scanqa'], qa_score_per_caption, 'scanqa')
                _log_to_disk(args, oc_message, object_caption_corpus, candidates['object_caption'], oc_score_per_caption, 'object_caption')
                _log_to_disk(args, sc_message, scene_caption_corpus, candidates['scene_caption'], sc_score_per_caption, 'scene_caption')
                
            if args.distributed:
                torch.cuda.synchronize()

        # Add testing results to TensorBoard
        if val_writer is not None:
            val_writer.add_scalar('Metric/QA', qa_score_per_caption['cider'], epoch)
            val_writer.add_scalar('Metric/OC', oc_score_per_caption['cider'], epoch)
            val_writer.add_scalar('Metric/SC', sc_score_per_caption['cider'], epoch)
        '''
        
    return Acc_Metric(0.)



def _log_to_disk(args, message, corpus, candidates, score_per_caption, task_name, epoch):
    with open(os.path.join(args.experiment_path, f"{task_name}_{epoch}_qa_scores.json"), "w") as f: 
                json.dump(message, f)
            
    with open(os.path.join(args.experiment_path, f"{task_name}_{epoch}_qa_corpus_val.json"), "w") as f: 
        json.dump(corpus, f, indent=4)
    
    with open(os.path.join(args.experiment_path, f"{task_name}_{epoch}_qa_pred_val.json"), "w") as f:
        json.dump(candidates, f, indent=4)
    
    with open(os.path.join(args.experiment_path, f"{task_name}_{epoch}_qa_pred_gt_val.json"), "w") as f:
        pred_gt_val = {}
        for scene_object_id, scene_object_id_key in enumerate(candidates):
            pred_gt_val[scene_object_id_key] = {
                'pred': candidates[scene_object_id_key],
                'gt': corpus[scene_object_id_key],
                'score': {
                    'bleu-1': score_per_caption['bleu-1'][scene_object_id],
                    'bleu-2': score_per_caption['bleu-2'][scene_object_id],
                    'bleu-3': score_per_caption['bleu-3'][scene_object_id],
                    'bleu-4': score_per_caption['bleu-4'][scene_object_id],
                    'CiDEr': score_per_caption['cider'][scene_object_id],
                    'rouge': score_per_caption['rouge'][scene_object_id],
                    'meteor': score_per_caption['meteor'][scene_object_id]
                }
            }
        json.dump(pred_gt_val, f, indent=4)

def test_net():
    pass