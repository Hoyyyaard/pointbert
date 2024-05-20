import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
import math
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train' if not hasattr(config.others, 'shuffle') else config.others.shuffle
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                t_mul=1,
                lr_min=1e-6,
                decay_rate=0.1,
                warmup_lr_init=1e-6,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return optimizer, scheduler

def build_llm_opti_sche(base_model, config, train_loader, finetune=False):
    
    if not finetune:
        for n, p in base_model.module.llm.named_parameters():
            p.requires_grad = False
        for n, p in base_model.module.encoder.named_parameters():
            p.requires_grad = False
    else:
        for n, p in base_model.module.llm.named_parameters():
            p.requires_grad = True
        for n, p in base_model.module.encoder.named_parameters():
            p.requires_grad = False
    
    opti_params = filter(lambda p: p.requires_grad, base_model.parameters())
    
    opti_config = config.optimizer
    # if opti_config.type == 'AdamW':
    #     def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    #         decay = []
    #         no_decay = []
    #         for name, param in model.module.named_parameters():
    #             if not param.requires_grad:
    #                 continue  # frozen weights
    #             if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
    #                 # print(name)
    #                 no_decay.append(param)
    #             else:
    #                 decay.append(param)
    #         return [
    #             {'params': no_decay, 'weight_decay': 0.},
    #             {'params': decay, 'weight_decay': weight_decay}]
    #     param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
    #     optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    # elif opti_config.type == 'Adam':
    #     optimizer = optim.Adam(opti_params, **opti_config.kwargs)
    # elif opti_config.type == 'SGD':
    #     optimizer = optim.SGD(opti_params, nesterov=True, **opti_config.kwargs)
    # else:
    #     raise NotImplementedError()

    sche_config = config.scheduler
    # if sche_config.type == 'LambdaLR':
    #     scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    # elif sche_config.type == 'CosLR':
    #     scheduler = CosineLRScheduler(optimizer,
    #             t_initial=sche_config.kwargs.epochs,
    #             t_mul=1,
    #             lr_min=1e-6,
    #             decay_rate=0.1,
    #             warmup_lr_init=1e-6,
    #             warmup_t=sche_config.kwargs.initial_epochs,
    #             cycle_limit=1,
    #             t_in_epochs=True)
    # elif sche_config.type == 'StepLR':
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    # elif sche_config.type == 'function':
    #     scheduler = None
    # else:
    #     raise NotImplementedError()
    
    # if config.get('bnmscheduler') is not None:
    #     bnsche_config = config.bnmscheduler
    #     if bnsche_config.type == 'Lambda':
    #         bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
    #     scheduler = [scheduler, bnscheduler]
    
    total_epochs = sche_config.kwargs.epochs
    warmup_ratio = opti_config.kwargs.warmup_ratio  # Warmup ratio of 3%
    initial_lr = opti_config.kwargs.lr  # Initial learning rate
    warmup_lr = 1e-6

    # Define optimizer
    eps = 1e-4 if finetune else 1e-8
    optimizer = optim.Adam(opti_params, lr=initial_lr, weight_decay=opti_config.kwargs.weight_decay, eps=eps)  # No weight decay

    # Define scheduler with warmup
    total_steps = len(train_loader) * total_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print("warmup_steps", warmup_steps)
    
    # Define lambda function for warmup scheduler
    def lr_lambda(step):
        if step < warmup_steps:
            return warmup_lr + step * (initial_lr - warmup_lr) / warmup_steps
        else:
            return 0.5 * (math.cos((step - warmup_steps) / (total_steps - warmup_steps) * math.pi) + 1) * initial_lr

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler, warmup_steps

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    msg = base_model.load_state_dict(base_ckpt, strict = False)
    print_log(msg)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def save_checkpoint_pretrain_llm(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        weight_ckpt = base_model.module.state_dict() if args.distributed else base_model.state_dict()
        parameter_names = list(weight_ckpt.keys())
        for name in parameter_names:
            if name.find('llm.') != -1:
                weight_ckpt.pop(name)
        torch.save({
                    'base_model' : weight_ckpt,
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict = True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return 