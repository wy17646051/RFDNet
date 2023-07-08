import logging
import pickle
from pathlib import Path
from typing import Any, List, Tuple

from omegaconf import OmegaConf

from utils.distributed.comm import main_process_only

LOG = logging.getLogger(__name__)


def update_resume_config(config_path: str, items: List[Tuple[str, Any]], merge: bool = True, force_add: bool = False):
    config_path = Path(config_path)
    if not config_path.exists():
        LOG.warning(f'The path of resume config `{config_path}` does not exist.')
        return

    with open(config_path, "rb") as file:
        config = pickle.load(file)

    for item in items:
        key, value = item
        OmegaConf.update(config, key, value, merge=merge, force_add=force_add)

    with open(config_path, "wb") as file:
        pickle.dump(config, file, protocol=4)
    LOG.info(f'Updating resume config in {config_path}.')


@main_process_only
def update_run_state(run_dir: str):
    run_dir = Path(run_dir)

    latest_wandb = list((run_dir / 'wandb'/ 'latest-run').rglob('*.wandb'))
    if len(latest_wandb) != 1:
        LOG.warning('Failed to save running state, current wandb id of run not recognized.')
        return 
    latest_id = str(latest_wandb[0].stem).split('-')[1]    

    update_resume_config(
        run_dir / '.hydra' / 'resume.pickle',
        [
            ('RESUME', str(run_dir / 'checkpoints' / 'checkpoint_last.pth')),
            ('WANDB.resume', latest_id)
        ], 
        force_add=True
    )
    LOG.info('Running status updated successfully.')

