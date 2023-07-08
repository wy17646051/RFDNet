from copy import deepcopy
from time import time
import torch.distributed as dist

import torch
import wandb

from utils.distributed import comm


class WandbStepWiseLogger:
    """
    Not verified on multi-machine distribution.
    """
    def __init__(self, project, run_name, output_dir, entity, resume=None, mode=None, **kwargs):
        self.is_dist = comm.is_distributed()
        self.is_logger = comm.is_main_process()
        
        if self.is_logger:
            wandb.init(
                project=project, name=run_name, dir=output_dir, entity=entity, 
                id=resume, resume='allow', mode=mode, **kwargs
            )
        self.meters = dict()
        # NOTE synchronize can be removed
        comm.synchronize()

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def _all_reduce_dict(self, input_dict, average=True):
        if not self.is_dist:
            return input_dict
            
        world_size = dist.get_world_size()
        if world_size == 1:
            return input_dict

        with torch.no_grad():
            names = []
            values = []
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(torch.as_tensor(input_dict[k]).to('cuda'))
            values = torch.stack(values)
            dist.all_reduce(values)
            if average:
                values /= world_size
            reduced_dict = {k: v for k, v in zip(names, values)}

        return reduced_dict

    def _wanb_log(self, info_dict, commit, reduce=True, **kwargs):
        if reduce:
            comm.synchronize()
            info_dict = self._all_reduce_dict(info_dict)
        if self.is_logger:
            wandb.log(info_dict, commit=commit, **kwargs)

    def logging(self, iterable, timing=True, commit=True, reduce=True):
        """
        Note that the local cache log of the previous iteration will be cleared before each iteration.

        Args:
            iterable (Iterable): Iterable object.
            timing (bool): whether to record and submit the time. Default to True.
            commit (bool): Whether to submit the log generated by the last iteration to the wandb server.
                Note that even if it is set to False, the remaining iterations will still submit 
                the log generated by the iteration to the wandb server. Default to True.
        """
        self.clear_log()

        time_iter_end = time_start = time()
      
        for idx, obj in enumerate(iterable):
            time_iter_start = time()
            expend_data_laoding = time_iter_start - time_iter_end
            yield obj

            expend_iter = time() - time_iter_start

            metrics = deepcopy(self.meters)
            if timing:
                metrics.update({
                    'expend_data_laoding': expend_data_laoding, 
                    'expend_iter': expend_iter
                })

            commit_iter = False if idx + 1 == len(iterable) else True
            self._wanb_log(metrics, commit=commit_iter, reduce=reduce)

            time_iter_end = time()
        
        if timing:
            self._wanb_log({'expend_batch': time_iter_end - time_start}, commit=False, reduce=reduce)
        
        self._wanb_log({}, commit=commit, reduce=False)
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.meters[k] = torch.as_tensor(v).item()

    def log_once(self, commit=True, reduce=True, **kwargs):
        once_log = {}

        for k, v in kwargs.items():
            once_log[k] = torch.as_tensor(v).item()
        
        self._wanb_log(once_log, commit=commit, reduce=reduce)

    def manual_commit(self):
        self._wanb_log({}, commit=True, reduce=False)

    def save_state(self, *args, **kwargs):
        if self.is_logger:
            torch.save(*args, **kwargs)

    def clear_log(self):
        if self.is_dist:
            comm.synchronize()
        self.meters.clear()
