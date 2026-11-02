import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.kernels import get_gaussian_kernel2d


class ElasticTransform(nn.Module):
    """
    Add random elastic transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self, shift_scale: float = 1.0, kernel_size: int = 63, sigma: float = 32, align_corners: bool = False, mode: str = "bilinear"):
        super(ElasticTransform, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.align_corners = align_corners
        self.shift_scale = shift_scale
        self.mode = mode

    def forward(self, input):
        # generate noise
        batch_size, _, height, weight = input.shape
        noise = torch.rand(batch_size, 2, height, weight) * 2 - 1
        # print(torch.max(noise),torch.min(noise))
        noise=noise * self.shift_scale
        # elastic transform
        warped, disp = self.elastic_transform2d(input, noise)
        return warped, disp

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

        return warped, disp





# class ElasticTransformJoint(nn.Module):
#     """
#     Add random elastic transforms to a tensor image.
#     Most functions are obtained from Kornia, difference:
#     - gain the disp grid
#     - no p and same_on_batch
#     """

#     def __init__(self, shift_scale: float = 1.0, kernel_size: int = 63, sigma: float = 32, align_corners: bool = False, mode: str = "bilinear"):
#         super(ElasticTransform, self).__init__()
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#         self.align_corners = align_corners
#         self.shift_scale = shift_scale
#         self.mode = mode

#     def forward(self, input1,input2):
#         # generate noise
#         batch_size, _, height, weight = input.shape
#         noise = torch.rand(batch_size, 2, height, weight) * 2 - 1
#         # print(torch.max(noise),torch.min(noise))
#         noise = noise * self.shift_scale
#         # elastic transform
#         warped1, warped2, disp = self.elastic_transform2d(input1,input2, noise)
        
#         return warped1, warped2, disp

#     def elastic_transform2d(self, image1: torch.Tensor, image2: torch.Tensor, noise: torch.Tensor):
#         # if not isinstance(image1, torch.Tensor):
#         #     raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

#         # if not isinstance(noise, torch.Tensor):
#         #     raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

#         # if not len(image.shape) == 4:
#         #     raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

#         # if not len(noise.shape) == 4 or noise.shape[1] != 2:
#         #     raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

#         # unpack hyper parameters
#         kernel_size = self.kernel_size
#         sigma = self.sigma
#         align_corners = self.align_corners
#         mode = self.mode
#         device = image1.device

#         # Get Gaussian kernel for 'y' and 'x' displacement
#         # kernel_x: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]
#         # kernel_y: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]
#         kernel_x: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
#         kernel_y: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))

#         # Convolve over a random displacement matrix and scale them with 'alpha'
#         disp_x: torch.Tensor = noise[:, :1].to(device)
#         disp_y: torch.Tensor = noise[:, 1:].to(device)

#         disp_x = kornia.filters.filter2d(disp_x, kernel=kernel_y, border_type="constant")
#         disp_y = kornia.filters.filter2d(disp_y, kernel=kernel_x, border_type="constant")

#         # stack and normalize displacement
#         disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

#         # Warp image based on displacement matrix
#         b, c, h, w = image1.shape
#         grid = kornia.utils.create_meshgrid(h, w, device=image1.device).to(image1.dtype)
#         warped1 = F.grid_sample(image1, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)
#         warped2 = F.grid_sample(image2, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)

#         return warped1, warped2, disp
