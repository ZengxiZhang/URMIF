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
import cv2
class HomoElasticTransform(nn.Module):
    """
    Add random affine transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self,trans=0.15, shift=0.15,hw=256, shift_scale: float = 1.0, kernel_size: int = 63, sigma: float = 32, align_corners: bool = False, mode: str = "bilinear"):
        # super(HomoTransform, self).__init__(trans=0.15, shift=0.15,hw=256)
        super().__init__()
        # super(HomoTransform, self).__init__(trans=0.15, shift=0.15,hw=256)
        self.trans = trans
        self.shift = shift
        self.hw = hw
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.align_corners = align_corners
        self.shift_scale = shift_scale
        self.mode = mode
        

    def forward(self, input_ir, input_vis, need_mask=False):
        x_trans = rd.randint(-int(0.15 * self.hw), int(0.15 * self.hw))
        y_trans = rd.randint(-int(0.15 * self.hw), int(0.15 * self.hw))
        x_off = np.array([0,0,0,0])
        y_off = np.array([0,0,0,0])
        for i in range(len(x_off)):
            x_off[i] = rd.randint(int(-0.15*256), int(0.15*256))
        for i in range(len(y_off)):
            y_off[i] = rd.randint(int(-0.15*256), int(0.15*256))
        _, _, h, w = input_ir.shape
        # x_off=x_off*0
        # y_off=y_off*0
        # x_trans=0
        # y_trans=0
        while True:
            x_ini = rd.randint( -(x_trans + min(x_off[0],x_off[2]))       if (x_trans + min(x_off[0],x_off[2]))<0 else 1,
                            w-(x_trans + max(x_off[1],x_off[3]))-256-1 if (x_trans + max(x_off[1],x_off[3]))>0 else w-256-1)
            y_ini = rd.randint( -(y_trans + min(y_off[0],y_off[1]))       if (y_trans + min(y_off[0],y_off[1]))<0 else 1,
                            h-(y_trans + max(y_off[2],y_off[3]))-256-1 if (y_trans + max(y_off[2],y_off[3]))>0 else h-256-1)
            x_ini =y_ini= 100
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

        # print("########################################################################")
        H1 = kornia.geometry.homography.find_homography_dlt(input3_loc, input2_loc).to(input_ir.device)
        H2 = kornia.geometry.homography.find_homography_dlt(input1_loc - np.array([x_ini,y_ini]), input3_loc - np.array([x_ini,y_ini])).to(input_ir.device)
        # H1 = torch.eye(3)[None].to(input_ir.device)
        # H2 = torch.eye(3)[None].to(input_ir.device)
        # print(input3_loc)
        # print(input2_loc)
        # generate noise
        batch_size, _, height, weight = input_vis.shape
        noise = torch.rand(batch_size, 2, height, weight) * 2 - 1
        # print(torch.max(noise),torch.min(noise))
        noise = noise * self.shift_scale
        print(self.shift_scale)
        # elastic transform
        img_grid = self._draw_grid(input_vis.squeeze().cpu().numpy(), 24)[None]
        # print(noise.shape)
        # noise[:,0] = noise[:,0]*0
        # print(input_vis.shape)
        # for h in range(0,input_vis.shape[2],50):
        #     input_vis[0,0,h:h+3]=torch.tensor(0)
        #     input_vis[0,1,h:h+3]=torch.tensor(0)
        #     input_vis[0,2,h:h+3]=torch.tensor(255)
        # for h in range(0,input_vis.shape[3],50):
        #     input_vis[0,0,:,h:h+3]=torch.tensor(0)
        #     input_vis[0,1,:,h:h+3]=torch.tensor(0)
        #     input_vis[0,2,:,h:h+3]=torch.tensor(255)
        input_vis2_elastic, disp = self.elastic_transform2d(input_vis, noise)
        input_ir2_elastic, disp = self.elastic_transform2d(input_ir, noise)
        img_grid_elastic, _    = self.elastic_transform2d(img_grid, noise)

        # grid = kornia.utils.create_meshgrid(h, w, device=img_grid.device).to(img_grid.dtype)
        # img_grid_back = torch.nn.functional.grid_sample(img_grid_elastic, new_grid, padding_mode='zeros', align_corners=False)
        # img_grid_back = F.grid_sample(img_grid_elastic, (grid - disp).clamp(-1, 1), align_corners=self.align_corners, mode=self.mode)
        # img_grid = img_grid[0].permute(1,2,0).detach().cpu().numpy()*255
        # img_grid_back = img_grid_back[0].permute(1,2,0).detach().cpu().numpy()*255
        # img_grid_elastic = img_grid_elastic[0].permute(1,2,0).detach().cpu().numpy()*255
        # img_grid_elastic2 = img_grid_elastic2[0].permute(1,2,0).detach().cpu().numpy()*255
        # save = np.concatenate((img_grid, img_grid_elastic, img_grid_elastic2,img_grid_back),0)
        # cv2.imwrite('save.png',save)
        # print("h")
        # time.sleep(1000)
        # print(H1)
        # print(input_vis_elastic.shape,H1.shape)
        input_ir2_elastic_warped = self.warp_perspective(input_ir2_elastic, H1.float(), dsize=[h, w])
        input_vis2_elastic_warped = self.warp_perspective(input_vis2_elastic, H1.float(), dsize=[h, w])
        img_grid_elastic_warped = self.warp_perspective(img_grid_elastic, H1.float(), dsize=[h, w])
        # print(input_vis_warped.shape,input_ir.shape)
        
        ir1 = input_ir[:,:, y_ini:y_ini+self.hw, x_ini:x_ini+self.hw]
        vis1 = input_vis[:,:, y_ini:y_ini+self.hw, x_ini:x_ini+self.hw]
        # vis_gt = input_vis[:,:, y_ini:y_ini+self.hw, x_ini:x_ini+self.hw]
        vis2 = input_vis2_elastic_warped[:,:, y_ini+y_trans:y_ini+y_trans+self.hw, x_ini+x_trans:x_ini+x_trans+self.hw]
        ir2 = input_ir2_elastic_warped[:,:, y_ini+y_trans:y_ini+y_trans+self.hw, x_ini+x_trans:x_ini+x_trans+self.hw]
        img_grid_elastic_warped = img_grid_elastic_warped[:,:, y_ini+y_trans:y_ini+y_trans+self.hw, x_ini+x_trans:x_ini+x_trans+self.hw]
        # print(img_grid_elastic_warped.shape)
        # print(torch.max(img_grid_elastic_warped), torch.min(img_grid_elastic_warped))
        # time.sleep(1000)
        # _,_,h,w = disp.shape
        disp = disp[:, y_ini:y_ini+self.hw, x_ini:x_ini+self.hw,:]
        # disp = torch.ones_like(disp)
        # disp = torch.zeros_like(disp)
        # disp[:,0]= disp[:,0]*0
        # _,_,h,w=disp.shape
        # print(disp.shape)
        disp[:,:,:,0] = disp[:,:,:,0].clone()/self.hw*w
        disp[:,:,:,1] = disp[:,:,:,1].clone()/self.hw*h
        disp = -disp
        # disp[:,0] = disp[:,0] * (self.hw) /w + (x_ini/(self.hw) - 1)
        # disp[:,1] = disp[:,1] * (self.hw) /h + (y_ini/(self.hw) - 1)
        # disp = disp[:, y_ini:y_ini+self.hw, x_ini:x_ini+self.hw,:]
        # disp = disp[:, y_ini:y_ini+self.hw, x_ini:x_ini+self.hw,:]
        # disp[:,0]= disp[:,0]/self.hw*w
        # disp[:,1]= disp[:,1]/self.hw*h
        # print(((self.hw)*1. / w) * 2 - 1)
        # disp[..., 0] = (disp[..., 0] - x_ini / (w / 2. - 1)) / ((self.hw)*1. / w) * 2 - 1
        # disp[..., 1] = (disp[..., 1] - y_ini / (h / 2. - 1)) / ((self.hw)*1. / h) * 2 - 1
        # disp[:,0] = ((disp[:,0]-torch.min(disp[:,0]))/(torch.max(disp[:,0])-torch.min(disp[:,0])))*2-1
        # disp[:,1] = ((disp[:,1]-torch.min(disp[:,1]))/(torch.max(disp[:,1])-torch.min(disp[:,1])))*2-1
        # vis_gt = self.warp_perspective(vis, H2.float(), dsize=[self.hw, self.hw])
        # vis = self.warp_perspective(vis, H2.float(), dsize=[296, 534])
        # print(ir.shape, vis.shape, vis_gt.shape)
        
        if need_mask:
            mask = torch.ones_like(vis2)
            input_mask_warped = self.warp_perspective(mask, H2.float(), dsize=[self.hw, self.hw])
            return ir1, ir2, vis1, vis2, img_grid_elastic_warped, H2.float(), input3_loc - input1_loc, input_mask_warped
        else:
            return ir1, ir2, vis1, vis2, img_grid_elastic_warped, H2.float(), input3_loc - input1_loc, disp

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
    
    def elastic_transform2d(self, image: torch.Tensor, noise: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

        if not len(noise.shape) == 4 or noise.shape[1] != 2:
            raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

        # unpack hyper parameters
        kernel_size = self.kernel_size
        sigma = self.sigma
        align_corners = self.align_corners
        mode = self.mode
        device = image.device

        # Get Gaussian kernel for 'y' and 'x' displacement
        # kernel_x: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]
        # kernel_y: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]
        kernel_x: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
        kernel_y: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))

        # Convolve over a random displacement matrix and scale them with 'alpha'
        disp_x: torch.Tensor = noise[:, :1].to(device)
        disp_y: torch.Tensor = noise[:, 1:].to(device)

        disp_x = kornia.filters.filter2d(disp_x, kernel=kernel_y, border_type="constant")
        disp_y = kornia.filters.filter2d(disp_y, kernel=kernel_x, border_type="constant")

        # stack and normalize displacement
        disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

        # Warp image based on displacement matrix
        b, c, h, w = image.shape
        grid = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
        warped = F.grid_sample(image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)
        # warped2 = F.grid_sample(warped, (grid - disp).clamp(-1, 1), align_corners=align_corners, mode=mode)
        # warped2 = torch.nn.functional.grid_sample(warped, (grid - disp).clamp(-1, 1), padding_mode='border', align_corners=False)
        # return warped,warped2, disp
        return warped, disp
    @staticmethod
    def _draw_grid(im_cv, grid_size: int = 10):
        # print(im_cv.shape)
        # print(type(im_cv))
        im_gd_cv = np.full_like(np.transpose(im_cv,(1,2,0)), 255.0)
        # im_gd_cv = cv2.cvtColor(im_gd_cv, cv2.COLOR_GRAY2BGR)
        color = (0, 0, 255)
        height, width, _ = im_gd_cv.shape
        for x in range(0, width - 1, grid_size):
            cv2.line(im_gd_cv, (x, 0), (x, height), color, 1, 1)
        for y in range(0, height - 1, grid_size):
            cv2.line(im_gd_cv, (0, y), (width, y), color, 1, 1)

        im_gd_ts = kornia.utils.image_to_tensor(im_gd_cv / 255.).type(torch.FloatTensor).cuda()
        return im_gd_ts









