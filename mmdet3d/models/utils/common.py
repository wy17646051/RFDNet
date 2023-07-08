from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule


def clip_sigmoid(x, eps=1e-4):
    """Sigmoid function for input feature.

    Args:
        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float, optional): Lower bound of the range to be clamped to.
            Defaults to 1e-4.

    Returns:
        torch.Tensor: Feature map after sigmoid.
    """
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class CBN2d(BaseModule):
    """ Conv2d + BacthNorm2d + NoLinear
    """
    
    def __init__(self, 
                 in_channels, out_channels: int, kernel_size=3, stride=1, padding=1, bias=True,
                 no_linear=nn.ReLU(), init_cfg: Optional[dict] = None):
        super(CBN2d, self).__init__(init_cfg)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.no_linear = no_linear

    def forward(self, inputs):
        outputs = self.bn(self.conv(inputs))
        
        if isinstance(self.no_linear, nn.Module):
            outputs = self.no_linear(outputs)

        return outputs


class MLP(BaseModule):
    """ Very simple multi-layer perceptron."""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, end_nolinear=False, 
        init_cfg: Optional[dict] = None
    ):
        super(MLP, self).__init__(init_cfg)
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self.end_nolinear = end_nolinear

    def forward(self, inputs):
        """
        Args:
            inputs: shape with :math:`(..., D)`
        
        Return:
            outputs: shape with :math:`(..., E)`
        """
        for i, layer in enumerate(self.layers):
            inputs = F.relu(layer(inputs)) if i < self.num_layers - 1 else layer(inputs)
        return F.relu(inputs) if self.end_nolinear else inputs
