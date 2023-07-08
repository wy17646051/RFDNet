from functools import partial
from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch_scatter import scatter


class PointsProjection:
    """ Projecte points to 2D view and projecte back.

    Args:
        mapping_fn (Callable): The function to map the coordinates of points to 2D view, which should have the following signature:
            `mapping_fn(coord: Tensor) -> Tuple[Tensor]` where argument `coord` is the coordinates of points with shape :math:`(P, 
            4)` that contain (batch_index, x, y, z) and the returned value is the tuple of coordinates of valid points in 2D view 
            with shape :math:`(P, 3)` that contain (batch_index, u, v), where u and v are the coordinates in 2D view as follows:

                    ----------->  u(x)
                    |
                    |
                    |
                    v  v(y)

            and mask of point coordinates in valid area with shape :math: `(P, )`.
        view_shape (List[int, int]): The shape of 2D view, which is a List of [height, width].
        cache_coord (bool, optional): Whether to cache the coordinates of points in 2D view when calling `points2view` function. 
            Defaults to True.
        reduce (str, optional): The reduce type of points in the same grid cell, should be one of ['sum', 'mul', 'mean', 'min', 
            'max']. Defaults to 'max'.
        reduced_expansion (int, optional): The point features in the grid specified by the coordinate and the grid within the 
            `reduced_expansion` of the coordinate will be reduced, when called `points2view` function. Defaults to 0.
    """

    def __init__(self, mapping_fn: Callable, view_shape: List[int], cache_coord: bool = True, reduce: str='max', reduced_expansion: int=0) -> None:
        if reduce not in ['sum', 'mul', 'mean', 'min', 'max']:
            raise ValueError(f'Invalid reduce type {reduce}, should be one of [sum, mul, mean, min, max].')

        self.mapping_fn = mapping_fn
        self.view_shape = view_shape
        self.cache_coord = cache_coord
        self.reduce = reduce

        if isinstance(reduced_expansion, int):
            reduced_expansion = [reduced_expansion for _ in range(4)]
        elif isinstance(reduced_expansion, list or tuple):
            if len(reduced_expansion) == 1:
                reduced_expansion = [reduced_expansion[0] for _ in range(4)]
            elif len(reduced_expansion) == 2:
                reduced_expansion = [reduced_expansion[0], reduced_expansion[1], reduced_expansion[0], reduced_expansion[1]]
            elif len(reduced_expansion) == 4:
                reduced_expansion = list(reduced_expansion)
            else:
                raise ValueError(f'Invalid reduced_expansion length {len(reduced_expansion)}, should be 1 or 2 or 4.')
        else:
            raise ValueError(f'Invalid reduced_expansion type {type(reduced_expansion)}, should be int or list or tuple.')
        self.reduced_expansion = reduced_expansion
    
    def _scatter(self, points: Tensor, view_coord: Tensor, batch_size: int, view_shape: List[int]) -> Tensor:
        """
        Param:
            points (Tensor): with shape :math:`(P_reduce, D)`, where P is points number, D is feature dimention.
            view_coord (Tensor): with shape :math:`(P_reduce, 3)`, that contain (batch_idx, u, v).
            batch_size (int): batch size.
            view_shape (List[int]): with shape :math:`(2, )`, that contain (h, w).
        
        Return:
            view (Tensor): with shape :math:`(N, C, H, W)` where N is batch size, H and W is shape of grid map. 
        """
        h_grid, w_grid = view_shape
        feature_dim = points.shape[-1]
        ex_left, ex_top, ex_right, ex_bottom = self.reduced_expansion
        
        expand_area = (ex_left + ex_right + 1) * (ex_top + ex_bottom + 1)
        expand_offset = torch.stack(torch.meshgrid([
            torch.arange(-ex_left, ex_right + 1, device=points.device),
            torch.arange(-ex_top, ex_bottom + 1, device=points.device)]
        , indexing='xy'), dim=-1).view(-1, 2)
        
        expand_points = points.unsqueeze(1).expand(-1, expand_area, -1).contiguous()  # P_reduce * expand_area * D
        expand_view_coord = view_coord.long().unsqueeze(1).repeat(1, expand_area, 1)  # P_reduce * expand_area * 3
        expand_view_coord[..., 1:] += expand_offset
        expand_view_coord[..., 1] = torch.clamp(expand_view_coord[..., 1], 0, w_grid - 1)
        expand_view_coord[..., 2] = torch.clamp(expand_view_coord[..., 2], 0, h_grid - 1)

        expand_points = expand_points.view(-1, feature_dim)  # (P_reduce * expand_area) * D
        n, u_floor, v_floor = expand_view_coord.view(-1, 3).unbind(-1)  # (P_reduce * expand_area)

        grid_coord_flatten = n * (h_grid * w_grid) + v_floor * w_grid + u_floor
        view = scatter(expand_points, grid_coord_flatten, 0, None, batch_size * h_grid * w_grid, self.reduce)  # P_reduce * D 

        return view.view(batch_size, h_grid, w_grid, feature_dim).permute(0, 3, 1, 2).contiguous()

    def _gather(self, view: Tensor, view_coord: Tensor, view_shape: List[int]) -> Tensor:
        """
        Param:
            view: with shape :math:`(N, C, H, W)` where N is batch size, H and W is shape of grid map.
            view_coord: with shape :math:`(P_reduce, 3)`, that contain (batch_idx, u, v).
            view_shape: with shape :math:`(2, )`, that contain (h, w).
        
        Return:
            points: with shape :math:`(P_reduce, D)`. 
        """
        h_grid, w_grid = view_shape
        n, u, v = view_coord.unbind(-1)
        n, u_floor, v_floor = n.long(), u.long(), v.long()
        u_floorp, v_floorp = u_floor + 1, v_floor + 1
        channels, h, w = view.shape[1:]
        
        view_ = view.new_zeros([*view.shape[:2], h+1, w+1])
        view_[..., :h, :w] = view
        view = view_
        h_grid, w_grid = h_grid + 1, w_grid + 1

        # (u, v) (u, v+) (u+ v) (u+, v+)
        bilinear_coord_flatten = n * (h_grid * w_grid) + torch.stack([v_floor, v_floorp, v_floor, v_floorp]) * w_grid + \
                                 torch.stack([u_floor, u_floor, u_floorp, u_floorp])  # 4 * N
        bilinear_weight_flatten = (1 - torch.abs(u - torch.stack([u_floor, u_floor, u_floorp, u_floorp]))) * \
                                  (1 - torch.abs(v - torch.stack([v_floor, v_floorp, v_floor, v_floorp])))  # 4 * N

        view = view.permute(0, 2, 3, 1).contiguous().view(-1, channels)[:, :, None].expand(-1, -1, 4)
        bilinear_coord_flatten = bilinear_coord_flatten.t()[:, None, :].expand(-1, channels, -1)
        bilinear_points = torch.gather(view, 0, bilinear_coord_flatten)  # N, C, 4
        
        points = bilinear_weight_flatten.t().unsqueeze(1) * bilinear_points  # N, C, 4
        points = points.sum(dim=-1)  # N, C

        return points

    @staticmethod
    def _bev_mapping_fn(coords: Tensor, bev_area: List[float], view_shape: List[int], eps: float=1.0) -> Tuple[Tensor, Tensor]:
        """ Convert the coordinates to bev coordinates.

        Args:
            coords (Tensor): points coordinates with shape :math:`(P, 4)` that contain (batch_index, x, y, z)
            bev_area (list[float]): The point cloud area of bev in format [x_min, y_min, x_max, y_max]
            view_shape (list[int]): The shape of the view in format [height, width]
            eps (float): Prevent coordinate out of bounds of `view_shape`

        Return:
            bev_coord (Tenosr): BEV coordinates in valid area with shape :math:`(P_valid, 3)`, that contain (batch_index, u, v)
            keep (Tensor): mask of points in valid area with shape :math: `(P, )`
        """
        h_bev, w_bev = view_shape
        x_min, y_min, x_max, y_max = bev_area

        n, x, y, _ = coords.unbind(-1)
        keep_x = torch.logical_and(x > x_min, x < x_max)
        keep_y = torch.logical_and(y > y_min, y < y_max)
        keep = torch.logical_and(keep_x, keep_y)

        u = (x - x_min) / (x_max - x_min) * w_bev
        v = (y - y_min) / (y_max - y_min) * h_bev
        u.clamp_(0, w_bev - eps)
        v.clamp_(0, h_bev - eps)

        bev_coord = torch.stack([n, u, v], dim=-1)[keep]
        return bev_coord, keep

    @staticmethod
    def _range_mapping_fn(coord: Tensor, vertical_fov: List[float], view_shape: List[int], eps: float=1):
        """Convert the coordinates to range coordinates.
        
        Param:
            coord (Tensor): points coordinates with shape :math:`(P, 4)` that contain (batch_index, x, y, z)
            vertical_fov (list[float]): Radians of point cloud vertical field with in format [min, max].
            view_shape (list[int]): The shape of the view in format [height, width]
            eps (float): Prevent coordinate out of bounds of `view_shape`

        Return:
            range_coord (Tensor): Range coordinate in valid area with shape :math:`(P_valid, 3)`, that contain (batch_index, u, v)
            keep (Tensor): mask of points in valid area with shape :math: `(P, )`
        """
        h_range, w_range = view_shape
        v_down, v_up = vertical_fov

        n, x, y, z = coord.unbind(-1)

        # r, theta, phi denote the distance, zenith and azimuth angle respectively
        r_sqr = x**2 + y**2 + z**2
        theta = torch.arcsin(z / torch.sqrt(r_sqr + 1e-8))
        phi = torch.atan2(y, x)
        keep = torch.logical_and(theta > v_down, theta < v_up)

        u = 0.5 * (1 - phi / torch.pi) * w_range
        v = (1 - (theta - v_down) / (v_up - v_down)) * h_range
        u.clamp_(0, w_range - eps)
        v.clamp_(0, h_range - eps)

        range_coord = torch.stack([n, u, v], dim=-1)[keep]
        return range_coord, keep

    @classmethod
    def get_mapping_fn(cls, type: str, **kwargs) -> Callable:
        """Get the mapping function for bev projection

        Args:
            type (str): The type of mapping function, should be one of ['bev', 'range', 'polar']
            **kwargs: The key word arguments for mapping function in format of `mapping_fn(coord, *args, **kwargs) -> Tuple[Tensor]`

        Return:
            mapping_fn: mapping function for projection
        """
        if type not in ['bev', 'range', 'polar']:
            raise ValueError('The type of mapping function should be one of [bev, range, polar]')

        mapping_fn = partial(getattr(cls, f'_{type}_mapping_fn'), **kwargs)
        return mapping_fn

    def points2view(self, points: Tensor, coords: Tensor, batch_size: int) -> Tensor:
        """
        Param:
            points (Tensor): Points with shape :math:`(P, D)`, where P is the number of points, D is feature dimention.
            coords (Tensor): points coordinates with shape :math:`(P, 4)` that contain (batch_index, x, y, z)
            batch_size (int): batch size. 
        
        Return:
            view: with shape :math:`(N, D, H, W)` where N is batch size, H and W is height and width of the view. 
        """
        assert points.shape[0] == coords.shape[0], 'The number of points and coordinates should be equal.'
        
        view_coord, keep = self.mapping_fn(coords)
        if self.cache_coord:
            self.view_coord = view_coord
            self.keep = keep

        view = self._scatter(points[keep], view_coord, batch_size, self.view_shape)
        return view

    def view2points(self, view: Tensor, view_coords: Tensor=None, keep: Tensor=None) -> Tensor:
        """
        Args:
            view (Tensor): with shape :math:`(N, C, H, W)` where N is batch size, C is feature dimention, H and W is height and 
                width of the view.
            view_coords (Tensor): Valid points coordinates with shape :math:`(P_valid, 3)` that contain (batch_index, u, v), if 
                None, use the cached view_coords.
            keep (Tensor): Mask of points in valid area with shape :math: `(P, )`, if None, use the cached keep.

        Return:
            points (Tensor): Points with shape :math:`(P, C)`, where P is the number of points.
        """
        if view_coords is None and keep is None:
            view_coords = getattr(self, 'view_coord', None)
            keep = getattr(self, 'keep', None)
            if view_coords is None or keep is None:
                raise ValueError('The view_coords and keep should be provided if not cached.')

        elif view_coords is None or keep is None:
            raise ValueError('view_coords and keep should be both None or not None.')
        
        valid_points = self._gather(view, view_coords, self.view_shape)

        points = valid_points.new_zeros((keep.shape[0], valid_points.shape[1]))
        points[keep] = valid_points

        return points
