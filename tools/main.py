import logging
import os
import random
import sys
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict
from mmcv.parallel import MMDistributedDataParallel, collate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.getcwd())

from tools.engine import eval_engine, train_engine
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from utils.common.checkpoint import save_ckpt
from utils.distributed import comm, init_distributed_mode
from utils.hydra.rerun import update_run_state
from utils.training import optim
from utils.wandb import WandbStepWiseLogger

LOG = logging.getLogger(__name__)


def fixed_seed(seed, deterministic):
    rank = comm.get_rank()
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False

    LOG.info(f'Deterministic training with seed {seed} in rank {rank}.')


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def run(cfg: DictConfig):

    # Init distribution training.
    init_distributed_mode(
        **OmegaConf.to_container(cfg.DIST)
    )

    # Fixed random seed.
    fixed_seed(cfg.SEED, cfg.CUDNN_DETERMINISTIC)

    # Initial WandB logger
    logger = WandbStepWiseLogger(
        **OmegaConf.to_object(cfg.WANDB)
    )

    ############ Build Dataset & Dataloader ############ 
    dataset_train = build_dataset(OmegaConf.to_container(cfg.DATASET.TRAIN))
    dataset_val = build_dataset(OmegaConf.to_container(cfg.DATASET.VAL))
    """
    cfg_dataset_val = OmegaConf.to_container(cfg.DATASET.VAL)
    for i in range(len(cfg_dataset_val['pipeline'])):
        if cfg_dataset_val['pipeline'][i]['type'] == 'MultiScaleFlipAug3D':
            cfg_dataset_val['pipeline'][i]['img_scale'] = tuple(cfg_dataset_val['pipeline'][i]['img_scale'])
    dataset_val = build_dataset(cfg_dataset_val, dict(test_mode=True))
    """

    sampler_train = DistributedSampler(dataset_train) if comm.is_distributed() else RandomSampler(dataset_train)
    batch_sampler_train = BatchSampler(sampler_train, cfg.TRAINING.batch_size, drop_last=True)
    data_loader_train = DataLoader(
        dataset_train, 
        batch_sampler=batch_sampler_train, 
        collate_fn=partial(collate, samples_per_gpu=cfg.TRAINING.sample_size), 
        num_workers=cfg.TRAINING.num_workers
    )

    sampler_val = DistributedSampler(dataset_val, shuffle=False) if comm.is_distributed() else SequentialSampler(dataset_val)
    data_loader_val = DataLoader(
        dataset_val, 
        batch_size=1, 
        sampler=sampler_val, 
        drop_last=False, 
        collate_fn=partial(collate, samples_per_gpu=1), 
        num_workers=cfg.TRAINING.num_workers
    )

    ############ Build Model ############
    model = build_model(EasyDict(OmegaConf.to_container(cfg.MODEL)))
    model.init_weights()
    if cfg.TRAINING.get('pretrained', None) is not None:
        model.load_state_dict(torch.load(cfg.TRAINING.pretrained, map_location='cpu')['state_dict'])
        print(f'Load pretrained model from {cfg.TRAINING.pretrained}')
    # add an attribute for visualization convenience
    model.CLASSES = dataset_train.CLASSES
    LOG.info(f'Model: \n{model}')
    model = model.cuda()
    model = comm.distributed_only(torch.nn.SyncBatchNorm.convert_sync_batchnorm, default=model)(model)
    
    # model = comm.distributed_only(torch.nn.parallel.DistributedDataParallel, default=model)(
    #     model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
    # )
    model = MMDistributedDataParallel(
        model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, find_unused_parameters=True
    )
    model_wo_ddp = model.module if comm.is_distributed() else model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info(f'Total number of params: {n_parameters}')
    
    ############ Build Optimizer & Scheduler ############
    optimizer_cfg = EasyDict(OmegaConf.to_container(cfg.OPTIMIZER))
    optimizer = getattr(optim, optimizer_cfg.pop('type'))(
        params = model.module.parameters() if comm.is_distributed() else model.parameters(), 
        **optimizer_cfg
    )
    scheduler_cfg = EasyDict(OmegaConf.to_container(cfg.SCHEDULER))
    scheduler = getattr(optim.lr_scheduler, scheduler_cfg.pop('type'))(
        optimizer=optimizer, **scheduler_cfg
    )

    ############ Resume ############
    start_epoch = 0 
    if 'RESUME' in cfg:
        checkpoint = torch.load(cfg.RESUME, map_location='cpu')
        model_wo_ddp.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        LOG.info(f'Resume successed with checkpoint \"{cfg.RESUME}\".')

    LOG.info('Training start.')
    best_metrics = -1
    eval_intervals = cfg.TRAINING.get('eval_intervals', 1)
    for epoch in range(start_epoch, cfg.TRAINING.max_epoch):
        comm.distributed_only(sampler_train.set_epoch)(epoch)     

        train_engine(model, data_loader_train, optimizer, epoch, logger, cfg.TRAINING.grad_clip)
        scheduler.step()

        metrics = -1
        if (epoch+1) % eval_intervals == 0:
            metrics = eval_engine(model, data_loader_val, epoch, logger)

        save_ckpt({
            'state_dict': model_wo_ddp.state_dict(),
            'epoch': epoch+1,
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict()
        }, Path(cfg.WANDB.output_dir) / 'checkpoints', metrics > best_metrics, epoch=epoch)
        best_metrics = metrics if metrics > best_metrics else best_metrics
        update_run_state(cfg.WANDB.output_dir)

    LOG.info('Training end.')


if __name__ == '__main__':
    run()
