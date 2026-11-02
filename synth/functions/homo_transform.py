from typing import Tuple
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.kernels import get_gaussian_kernel2d
import time
from kornia.geometry.conversions import normalize_homography
from kornia.utils.helpers import _torch_inverse_cast
from kornia.utils import create_meshgrid
from kornia.geometry.linalg import transform_points
import random as rd
import numpy as np

class HomoTransform(nn.Module):
    """
    Add random affine transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self,trans=0.15, shift=0.15,hw=256):
        # super(HomoTransform, self).__init__(trans=0.15, shift=0.15,hw=256)
        super().__init__()
        # super(HomoTransform, self).__init__(trans=0.15, shift=0.15,hw=256)
        self.trans = trans
        self.shift = shift
        self.hw = hw
        

    def forward(self, input_ir, input_vis, need_mask=False):
        # generate noise
        batch_size, _, height, weight = input_ir.shape
        noise = torch.rand(batch_size, 2, height, weight) * 2 - 1
        # print(torch.max(noise),torch.min(noise))
        # noise=noise * self.shift_scale
        # elastic transform
        # warped, disp = self.elastic_transform2d(input, noise)

        x_trans = rd.randint(-int(0.15 * self.hw), int(0.15 * self.hw))
        y_trans = rd.randint(-int(0.15 * self.hw), int(0.15 * self.hw))
        x_off = np.array([0,0,0,0])
        y_off = np.array([0,0,0,0])
        for i in range(len(x_off)):
            x_off[i] = rd.randint(int(-0.15*256), int(0.15*256))
        for i in range(len(y_off)):
            y_off[i] = rd.randint(int(-0.15*256), int(0.15*256))
        # print(input_ir.shape)
        _, _, h, w = input_ir.shape
        # x_ini = rd.randint( -(x_trans + np.min(x_off)) if (x_trans + np.min(x_off))<0 else 0, w-(x_trans + np.max(x_off))-256 if (x_trans + np.max(x_off))>0 else w-256)
        # y_ini = rd.randint( -(y_trans + np.min(y_off)) if (y_trans + np.min(y_off))<0 else 0, h-(y_trans + np.max(y_off))-256 if (y_trans + np.max(y_off))>0 else h-256)
        # print("")
        # print(w,h)

        # print((y_trans + min(y_off[0],y_off[1])),(y_trans + max(y_off[2],y_off[3])))
        # print( -(y_trans + min(y_off[0],y_off[1]))       if (y_trans + min(y_off[0],y_off[1]))<0 else 0,h-(y_trans + max(y_off[2],y_off[3]))-256-1 if (y_trans + max(y_off[2],y_off[3]))>0 else h-256-1)
        while True:
            x_ini = rd.randint( -(x_trans + min(x_off[0],x_off[2]))       if (x_trans + min(x_off[0],x_off[2]))<0 else 1,
                            w-(x_trans + max(x_off[1],x_off[3]))-256-1 if (x_trans + max(x_off[1],x_off[3]))>0 else w-256-1)
            y_ini = rd.randint( -(y_trans + min(y_off[0],y_off[1]))       if (y_trans + min(y_off[0],y_off[1]))<0 else 1,
                            h-(y_trans + max(y_off[2],y_off[3]))-256-1 if (y_trans + max(y_off[2],y_off[3]))>0 else h-256-1)
            
            input1_loc = np.zeros((4,2))
            input2_loc = np.zeros((4,2))
            input3_loc = np.zeros((4,2))

            input1_loc[0] = [x_ini          , y_ini          ]
            input1_loc[1] = [x_ini + self.hw, y_ini          ]
            input1_loc[2] = [x_ini          , y_ini + self.hw]
            input1_loc[3] = [x_ini + self.hw, y_ini + self.hw]

            input2_loc[0] = [input1_loc[0,0] + x_trans, input1_loc[0,1] + y_trans]
            input2_loc[1] = [input1_loc[1,0] + x_trans, input1_loc[1,1] + y_trans]
            input2_loc[2] = [input1_loc[2,0] + x_trans, input1_loc[2,1] + y_trans]
            input2_loc[3] = [input1_loc[3,0] + x_trans, input1_loc[3,1] + y_trans]

            input3_loc[0] = [input2_loc[0,0] + x_off[0], input2_loc[0,1]  + y_off[0]]
            input3_loc[1] = [input2_loc[1,0] + x_off[1], input2_loc[1,1]  + y_off[1]]
            input3_loc[2] = [input2_loc[2,0] + x_off[2], input2_loc[2,1]  + y_off[2]]
            input3_loc[3] = [input2_loc[3,0] + x_off[3], input2_loc[3,1]  + y_off[3]]
            if np.min(input2_loc) > 0 and np.max(input2_loc[:,0]) < w and np.max(input2_loc[:,1]) < h:
                break
        input1_loc = torch.from_numpy(input1_loc).unsqueeze(0)
        input2_loc = torch.from_numpy(input2_loc).unsqueeze(0)
        input3_loc = torch.from_numpy(input3_loc).unsqueeze(0)
        # print(input1_loc)
        # print(input2_loc)
        # print(input3_loc)
        # print(w,h,x_ini,y_ini)
        # print(input1_loc)
        # print(input3_loc)
        # print("########################################################################")
        H1 = kornia.geometry.homography.find_homography_dlt(input3_loc, input2_loc).to(input_ir.device)
        # H2 = kornia.geometry.homography.find_homography_dlt(input1_loc - np.array([x_ini,y_ini]), input3_loc - np.array([x_ini,y_ini])).to(input_ir.device)
        H2 = kornia.geometry.homography.find_homography_dlt(input1_loc - np.array([x_ini,y_ini]), input3_loc - np.array([x_ini,y_ini])).to(input_ir.device)
        # print('')
        # print(input1_loc)
        # print(input1_loc - np.array([x_ini,y_ini]))
        # print(input3_loc - np.array([x_ini,y_ini]))
        # print(np.array([x_ini,y_ini]))
        # print(input1_loc - np.array([x_ini,y_ini]))
        input_vis_warped = self.warp_perspective(input_vis, H1.float(), dsize=[h, w])
        # print(input_vis_warped.shape,input_ir.shape)
        
        ir = input_ir[:,:, y_ini:y_ini+self.hw, x_ini:x_ini+self.hw]
        vis_gt = input_vis[:,:, y_ini:y_ini+self.hw, x_ini:x_ini+self.hw]
        vis = input_vis_warped[:,:, y_ini+y_trans:y_ini+y_trans+self.hw, x_ini+x_trans:x_ini+x_trans+self.hw]
        # print(vis.shape)
        vis_gt = self.warp_perspective(vis, H2.float(), dsize=[self.hw, self.hw])
        # vis = self.warp_perspective(vis, H2.float(), dsize=[296, 534])
        # print(ir.shape, vis.shape, vis_gt.shape)
        
        if need_mask:
            mask = torch.ones_like(vis)
            input_mask_warped = self.warp_perspective(mask, H2.float(), dsize=[self.hw, self.hw])
            return ir, vis, vis_gt, H2.float(), input3_loc-input1_loc, input_mask_warped
        else:
            return ir, vis, vis_gt, H2.float(), input3_loc-input1_loc

    def warp_perspective(
        self,
        src: torch.Tensor,
        M: torch.Tensor,
        dsize: Tuple[int, int],
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: bool = True,
        fill_value: torch.Tensor = torch.zeros(3),  # needed for jit
        return_grid=False
    ) -> torch.Tensor:
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"Input src type is not a torch.Tensor. Got {type(src)}")

        if not isinstance(M, torch.Tensor):
            raise TypeError(f"Input M type is not a torch.Tensor. Got {type(M)}")

        if not len(src.shape) == 4:
            raise ValueError(f"Input src must be a BxCxHxW tensor. Got {src.shape}")

        if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
            raise ValueError(f"Input M must be a Bx3x3 tensor. Got {M.shape}")

        # fill padding is only supported for 3 channels because we can't set fill_value default
        # to None as this gives jit issues.
        if padding_mode == "fill" and fill_value.shape != torch.Size([3]):
            raise ValueError(f"Padding_tensor only supported for 3 channels. Got {fill_value.shape}")

        B, _, H, W = src.size()
        h_out, w_out = dsize

        # we normalize the 3x3 transformation matrix and convert to 3x4
        dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), (h_out, w_out))  # Bx3x3

        src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3

        # this piece of code substitutes F.affine_grid since it does not support 3x3
        grid = (
            create_meshgrid(h_out, w_out, normalized_coordinates=True, device=src.device).to(src.dtype).repeat(B, 1, 1, 1)
        )
        grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

        if padding_mode == "fill":
            return self._fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value)
        if not return_grid:
            return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)
        else:
            return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode),grid
        
    def _fill_and_warp(
        self,
        src: torch.Tensor,
        grid: torch.Tensor,
        mode: str,
        align_corners: bool,
        fill_value: torch.Tensor,
    ) -> torch.Tensor:
        ones_mask = torch.ones_like(src)
        fill_value = fill_value.to(ones_mask)[None, :, None, None]  # cast and add dimensions for broadcasting
        inv_ones_mask = 1 - F.grid_sample(ones_mask, grid, align_corners=align_corners, mode=mode, padding_mode="zeros")
        inv_color_mask = inv_ones_mask * fill_value
        return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode="zeros") + inv_color_mask