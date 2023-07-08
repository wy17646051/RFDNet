import logging
from pathlib import Path
from typing import Any

import torch
from utils.distributed.comm import main_process_only

LOG = logging.getLogger(__name__)


@main_process_only
def save_ckpt(obj: Any, save_dir: str, is_best: bool, name: str='checkpoint', epoch: int=None):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    checkpoint_pathes = []
    if is_best:
        checkpoint_pathes.append(save_dir / f'{name}_best.pth')
        if epoch is not None:
            assert isinstance(epoch, int)
            checkpoint_pathes.append(save_dir / f'{name}_{epoch:03}.pth')
    checkpoint_pathes.append(save_dir / f'{name}_last.pth')

    for checkpoint_path in checkpoint_pathes:
        torch.save(obj, checkpoint_path)

    LOG.info(f'Successfully saved checkpoints in directory of {str(save_dir)}.')
