import math
import warnings

from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

class CosineAnnealingWarmUpLR(_LRScheduler):

    def __init__(self, optimizer, T_max, start_ratio, warm_up, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max - warm_up
        self.start_ratio = start_ratio
        self.warm_up = warm_up
        self.eta_min = eta_min
        super(CosineAnnealingWarmUpLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [base_lr * self.start_ratio for base_lr in self.base_lrs]

        elif self.last_epoch < self.warm_up:
            return [self.last_epoch / max(1, self.warm_up) * (base_lr - base_lr * self.start_ratio) for base_lr in self.base_lrs]

        elif self.last_epoch == self.warm_up:
            return self.base_lrs
            
        return [(1 + math.cos(math.pi * (self.last_epoch - self.warm_up) / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - self.warm_up - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
