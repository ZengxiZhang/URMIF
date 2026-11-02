# coding: utf-8

import argparse
import pathlib
import warnings
import os
import cv2
import numpy as np
import kornia
import torch
from torch import Tensor
import random as rd
from functions.homo_elastic_transform import HomoElasticTransform


class getDeformableImages:
    """principle: generate warped IR/VIS pairs with homo+elastic deformation."""

    def __init__(self, shift_scale=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Roadscene / nirscene 常用 0.7; Postdam 常用 0.04
        self.homo_elastic = HomoElasticTransform(
            trans=0.15, shift=0.15, hw=256,
            shift_scale=shift_scale, kernel_size=101, sigma=16
        )

    @torch.no_grad()
    def __call__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, dst: pathlib.Path, num, mode):
        ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix.lower() in ['.png', '.jpg', '.bmp', '.tif', '.tiff']]
        vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix.lower() in ['.png', '.jpg', '.bmp', '.tif', '.tiff']]
        assert len(ir_list) > 0 and len(vi_list) > 0, f'empty folder: ir={ir_folder}, vi={vi_folder}'

        os.makedirs(dst / mode, exist_ok=True)
        os.makedirs(dst / mode / 'elastic_disp', exist_ok=True)
        os.makedirs(dst / mode / 'homo_shift', exist_ok=True)

        for cout in range(num):
            print(cout)
            name_save = str(cout + 1).zfill(6) + '.png'
            name_disp = str(cout + 1).zfill(6) + '.npy'
            index = rd.randint(0, len(vi_list) - 1)
            ir_path = ir_list[index]
            vi_path = vi_list[index]

            ir = self.imread(ir_path, mini_wh=412).to(self.device)
            vi = self.imread(vi_path, mini_wh=412).to(self.device)
            ir1, ir2, vis1, vis2, warp_grid, H, shift, elastic_disp = self.homo_elastic(ir, vi, need_mask=False)

            np.save(dst / mode / 'elastic_disp' / name_disp, elastic_disp.data.cpu().numpy()[0])
            np.save(dst / mode / 'homo_shift' / name_disp, shift.detach().cpu().numpy()[0])
            self.imsave(ir1, dst / mode / 'ir_input1', name_save)
            self.imsave(ir2, dst / mode / 'ir_input2', name_save)
            self.imsave(vis1, dst / mode / 'vis_input1', name_save)
            self.imsave(vis2, dst / mode / 'vis_input2', name_save)
            self.imsave(warp_grid, dst / mode / 'warp_grid', name_save)

    @staticmethod
    def imread(path: pathlib.Path, mini_wh=None, unsqueeze=False):
        im_cv = cv2.imread(str(path))
        assert im_cv is not None, f'Image {str(path)} is invalid.'
        h, w, _ = im_cv.shape
        if mini_wh is not None:
            if h < mini_wh:
                scale = mini_wh / h
                h, w = mini_wh, int(scale * w)
            if w < mini_wh:
                scale = mini_wh / w
                w, h = mini_wh, int(scale * h)
            im_cv = cv2.resize(im_cv, (w, h))
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts[None]

    @staticmethod
    def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
        im_s = im_s if type(im_s) == list else [im_s]
        dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
        for im_ts, p in zip(im_s, dst):
            im_ts = im_ts.squeeze().cpu()
            p.parent.mkdir(parents=True, exist_ok=True)
            im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
            cv2.imwrite(str(p), im_cv)


def hyper_args():
    parser = argparse.ArgumentParser(description='Generating deformable registration bigfield data')
    # ---- 默认按 nirscene 本地路径；改 Postdam 时换下面注释即可 ----
    parser.add_argument('--ir', default='../nirscene_dataset/train/ir', type=pathlib.Path)
    parser.add_argument('--vi', default='../nirscene_dataset/train/rgb', type=pathlib.Path)
    parser.add_argument('--dst', default='../registration_nirscene_bigfield', type=pathlib.Path)
    # parser.add_argument('--ir', default='../Postdam_IVIF/ir', type=pathlib.Path)
    # parser.add_argument('--vi', default='../Postdam_IVIF/rgb', type=pathlib.Path)
    # parser.add_argument('--dst', default='../registration_postdam_bigfield', type=pathlib.Path)

    parser.add_argument('--num_train', default=10000, type=int)
    parser.add_argument('--num_test', default=500, type=int)
    # nirscene/roadscene: 0.7 ; postdam: 0.04
    parser.add_argument('--shift_scale', default=0.7, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = hyper_args()
    print(f'ir={args.ir}\nvi={args.vi}\ndst={args.dst}\nshift_scale={args.shift_scale}')
    data = getDeformableImages(shift_scale=args.shift_scale)
    data(ir_folder=args.ir, vi_folder=args.vi, dst=args.dst, num=args.num_train, mode='train')
    data(ir_folder=args.ir, vi_folder=args.vi, dst=args.dst, num=args.num_test, mode='test')
