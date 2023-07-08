"""
Adaptive handling of distributed and non-distributed situations.
"""

import logging
import functools

import torch
import torch.distributed as dist

LOG = logging.getLogger(__name__)


def _is_available_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if _is_available_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if _is_available_and_initialized():
        return dist.get_rank()
    return 0


def is_distributed() -> bool:
    return _is_available_and_initialized()
    

def is_main_process() -> bool:
    return get_rank() == 0


def synchronize() -> None:
    if not _is_available_and_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def main_process_only(func):
    """If distributed, only the main process executes the function.
    Note that it cannot be nested, otherwise the inner nested function with 
    wrapper `main_process_only` will only run by the main process and will be 
    blocked in the `synchronize()` position.

    Usage:
    @main_process_only
    def func(a, b): ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            ret = func(*args, **kwargs)
        synchronize()
        if is_main_process():
            return ret

    return wrapper


def distributed_only(func=None, default=None):
    """Only executes the function in distributed mode."""
    if func is None:
        return functools.partial(distributed_only, default=default)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_distributed():
            ret = func(*args, **kwargs)
            return ret
        else:
            return default

    return wrapper


def print_once(rank: int=0) -> None:
    """Disables printing when not in process of specified rank."""
    assert rank < get_world_size(), f'rank should smaller than world size {get_world_size()}'

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if get_rank() == rank or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
