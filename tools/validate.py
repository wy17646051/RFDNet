import logging
import os
import random
import sys
from functools import partial

from mmdet.core import encode_mask_results
import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict
from mmcv.parallel import MMDistributedDataParallel, collate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader,  SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from tools.engine import eval_engine
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from utils.distributed import comm, init_distributed_mode

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


@torch.no_grad()
def eval_engine(model, data_loader):
    model.eval()
    results = []
    for batch_sample in tqdm(data_loader):
        
        result = model(return_loss=False, rescale=True, **batch_sample)
        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in result]

        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (
                    bbox_results, encode_mask_results(mask_results))

        results.extend(result)

    results_dict = data_loader.dataset.evaluate(results)
    print(results_dict)



@hydra.main(config_path="../configs", config_name="config", version_base=None)
def run(cfg: DictConfig):

    # Init distribution training.
    init_distributed_mode(
        **OmegaConf.to_container(cfg.DIST)
    )

    # Fixed random seed.
    fixed_seed(cfg.SEED, cfg.CUDNN_DETERMINISTIC)

    ############ Build Dataset & Dataloader ############ 
    dataset_val = build_dataset(OmegaConf.to_container(cfg.DATASET.VAL))

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
    model.load_state_dict(torch.load(cfg.RESUME, map_location='cpu')['state_dict'])
    model.CLASSES = dataset_val.CLASSES
    LOG.info(f'Model: \n{model}')
    model = model.cuda()
    model = comm.distributed_only(torch.nn.SyncBatchNorm.convert_sync_batchnorm, default=model)(model)
    model = MMDistributedDataParallel(
        model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, find_unused_parameters=True
    )
    
    ############ Eval ############
    eval_engine(model, data_loader_val)

    
if __name__ == '__main__':
    run()
