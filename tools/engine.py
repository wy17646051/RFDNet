import pickle

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad
from tqdm import tqdm

from mmdet.core import encode_mask_results
from utils.distributed import comm


def train_engine(model, data_loader, optimizer, epoch, logger, grad_clip):
    model.train()

    pbar = tqdm(data_loader)
    pbar.set_description(f'Epoch [{epoch+1}]')
    for batch_sample in logger.logging(data_loader, commit=False):

        outputs = model.train_step(batch_sample, optimizer)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        
        if 'log_vars' in outputs:
            logger.update(**outputs['log_vars'], **{
                'epoch': epoch + 1,
                'learning_rate': optimizer.param_groups[0]["lr"]
            })

        optimizer.zero_grad()
        outputs['loss'].backward()

        if grad_clip is not None:
            params = list(filter(
                lambda p: p.requires_grad and p.grad is not None, model.parameters()
            ))
            if len(params) > 0:
                clip_grad.clip_grad_norm_(params, **grad_clip)

        optimizer.step()
        pbar.update()

def collect_results_gpu(result_part, size):
    rank, world_size = comm.get_rank(), comm.get_world_size()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    part_list = []
    for recv, shape in zip(part_recv_list, shape_list):
        part_list.append(
            pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    return ordered_results

@torch.no_grad()
def eval_engine(model, data_loader, epoch, logger):
    model.eval()
    results = []

    pbar = tqdm(data_loader)
    pbar.set_description(f'Evaluation [{epoch+1}]')
    for batch_sample in data_loader:
        
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

        pbar.update()

    results = collect_results_gpu(results, len(data_loader.dataset))
    results_dict = data_loader.dataset.evaluate(results)
    logger.log_once(**results_dict)
    
    result_value = results_dict['pts_bbox_NuScenes/NDS']
    return result_value


# CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run --nproc_per_node=2 main.py +experiments=centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nusmini
