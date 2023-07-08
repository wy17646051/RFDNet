import logging
import os

import torch
import torch.distributed as dist

from .comm import print_once, synchronize

LOG = logging.getLogger(__name__)


def init_distributed_mode(backend='nccl', url='env://', world_size=None, print_rank=0, **kwargs):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        if world_size is None:
            raise ValueError('The value of world_size is needed')
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        LOG.info('Training on non-distributed mode.')
        return

    torch.cuda.set_device(gpu)

    LOG.info(f'Distributed initialization (Rank: {rank}): {url}')

    dist.init_process_group(
        backend=backend, init_method=url, world_size=world_size, rank=rank, **kwargs
    )
    synchronize()

    if print_rank is not None:
        print_once(print_rank)
