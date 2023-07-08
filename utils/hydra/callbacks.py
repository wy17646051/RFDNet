import logging
import pickle
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from hydra.experimental.callback import Callback

LOG = logging.getLogger(__name__)


class DisableMultiRunCallback(Callback):

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        sys.exit(
            'Multi-run is not supported, due to the multi-run bug in current version 1.2 of hydra.'
        )


class UpdateWandbOutputDirCallback(Callback):

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        OmegaConf.update(config, 'WANDB.output_dir', config.hydra.run.dir, force_add=True)
        LOG.info(f'Update config: WANDB.output_dir={config.hydra.run.dir}')


class PickleResumeConfigCallback(Callback):

     def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:        
        output_dir = Path(config.hydra.runtime.output_dir) / config.hydra.output_subdir
        file_name = 'resume.pickle'

        with open(output_dir / file_name, "wb") as file:
            pickle.dump(config, file, protocol=4)

        LOG.info('Saving resume config in {}.'.format(str(output_dir / file_name)))
