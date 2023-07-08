import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import BACKBONES
from ..utils import MLP, CBN2d, PointsProjection


class ResidualDownSamplingBlock(BaseModule):

    def __init__(self, in_channels, out_channels, stride, init_cfg: Optional[dict] = None):
        super(ResidualDownSamplingBlock, self).__init__(init_cfg)

        self.cbn1 = CBN2d(in_channels, out_channels, stride=stride, no_linear=None)
        self.cbn2 = CBN2d(in_channels, out_channels, kernel_size=1, padding=0,
                          no_linear=nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

    def forward(self, inputs):
        outputs1 = self.cbn1(inputs)
        outputs2 = self.cbn2(inputs)
        
        return F.relu(outputs1 + outputs2)
    

class ResidualDownSamplingEncoder(BaseModule):

    def __init__(self, 
                 in_channels=64, 
                 encoder_channels=[32, 64, 128, 128], 
                 stride=[2, 2, 2, 2],
                 init_cfg: Optional[dict] = None):
        super(ResidualDownSamplingEncoder, self).__init__(init_cfg)
        assert len(encoder_channels) == len(stride)

        self.encoder = nn.ModuleList()
        for in_c, out_c, s in zip(([in_channels] + encoder_channels)[:-1], encoder_channels, stride):
            self.encoder.append(ResidualDownSamplingBlock(in_c, out_c, s))
    
    def forward(self, inputs):
        """
        Param:
            inputs: with shape of :math:`(N,C,H,W)`, where N is batch size
        """
        encoder_outputs = [inputs]
        for layer in self.encoder:
            encoder_outputs.append(layer(encoder_outputs[-1]))
        
        outputs = encoder_outputs[-1]
        outputs = F.interpolate(outputs, inputs.shape[-2:], mode='bilinear')

        return outputs


class PointDecorationBlock(BaseModule):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers=2, end_nolinear=False, 
        init_cfg: Optional[dict] = None
    ):
        super(PointDecorationBlock, self).__init__(init_cfg)
        self.output_dim = output_dim
        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers, end_nolinear)

    def forward(self, *args):
        outputs = torch.cat(list(args), dim=-1)
        return self.mlp(outputs)


@BACKBONES.register_module()
class RangeFeatureDecorationEncoder(BaseModule):
    def __init__(self, in_mlp, ds_encoder, out_mlp, 
                 pc_range, pc_fov, bev_shape, range_shape,
                 bev_ds_channels=[32, 64, 128, 128], bev_ds_stride=[2, 2, 2, 2], 
                 bev_expand=0, range_expand=0, **kwargs
    ):
        super(RangeFeatureDecorationEncoder, self).__init__()
        assert len(bev_ds_channels) == len(bev_ds_stride)

        self.in_mlp = MLP(**in_mlp)
        self.ds_encoder = ResidualDownSamplingEncoder(**ds_encoder)
        self.out_mlp = MLP(**out_mlp)

        pc_fov = (torch.tensor(pc_fov) / 180) * math.pi
        pc_fov = pc_fov.tolist()
        self.proj_range = PointsProjection(
            mapping_fn = PointsProjection.get_mapping_fn(
                type = 'range',
                vertical_fov = pc_fov,
                view_shape = range_shape,
                eps = 1
            ),
            view_shape = range_shape,
            cache_coord = True,
            reduce = 'max',
            reduced_expansion=range_expand
        )

        self.proj_bev = PointsProjection(
            mapping_fn = PointsProjection.get_mapping_fn(
                type = 'bev', 
                bev_area = pc_range, 
                view_shape = bev_shape,
                eps = 1.0
            ),
            view_shape = bev_shape,
            cache_coord = True,
            reduce = 'max',
            reduced_expansion=bev_expand
        )

        self.bev_ds_encoder = torch.nn.ModuleList()
        for in_c, out_c, s in zip(([out_mlp['output_dim']] + bev_ds_channels)[:-1], bev_ds_channels, bev_ds_stride):
            self.bev_ds_encoder.append(ResidualDownSamplingBlock(in_c, out_c, s))

    def forward(self, inputs, coord, batch_size):
        """
        Param:
            inputs: points with shape of :math:`(P, D)`, where P is point number, D is number of feature dimentions.
            coord: points coordination with shape :math:`(P, 4)`, that contain (batch_index, x, y, z).
            batch_size: batch size.

        Return:
            outputs: with shape :math:`(P, D_out)`, where D_out is output dimention of `PointFusion`
        """
        assert inputs.shape[0] == coord.shape[0]

        # first mlp and p2g
        mlp_outputs = self.in_mlp(inputs)

        inputs_range = self.proj_range.points2view(mlp_outputs, coord, batch_size)
        outputs_range = self.ds_encoder(inputs_range)
        outputs_range = self.proj_range.view2points(outputs_range)

        out_decoration = self.out_mlp(
            torch.cat([mlp_outputs, outputs_range], dim=-1)
        )
        inputs_bev = self.proj_bev.points2view(out_decoration, coord, batch_size)

        outputs = inputs_bev
        for layer in self.bev_ds_encoder:
            outputs = layer(outputs)

        return outputs
