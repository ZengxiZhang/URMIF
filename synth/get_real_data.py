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
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functions.homo_elastic_transform import HomoElasticTransform
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULTS = {'ir': 'RIVIR/IR', 'vi': 'RIVIR/Label', 'dst': 'registration_RIVIR_syn_real_10000', 'num_train': 10000, 'num_test': 0, 'shift_scale': 0.01, 'trans': 0.0, 'shift': 0.15, 'patch_size': 256, 'corner_x_ranges': [-0.4, -0.1, 0.1, 0.4, -0.4, -0.1, 0.1, 0.4], 'corner_y_ranges': [-0.3, 0.0, -0.3, 0.0, 0.3, 0.6, 0.3, 0.6], 'random_crop': True, 'crop_origin': [100, 100], 'source_short_side': 514, 'seed': 2026, 'workers': 8, 'device': 'cpu'}
for key in ('ir', 'vi', 'dst'):
    DEFAULTS[key] = PROJECT_ROOT / DEFAULTS[key]

class getDeformableImages:

    def __init__(self, shift_scale=0.7, trans=0.15, shift=0.15, patch_size=256, corner_x_ranges=None, corner_y_ranges=None, random_crop=False, crop_origin=(100, 100), source_short_side=None, device='auto'):
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        self.source_short_side = source_short_side
        self.homo_elastic = HomoElasticTransform(trans=trans, shift=shift, hw=patch_size, shift_scale=shift_scale, kernel_size=101, sigma=16, corner_x_ranges=corner_x_ranges, corner_y_ranges=corner_y_ranges, random_crop=random_crop, crop_origin=crop_origin)

    @torch.no_grad()
    def __call__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, dst: pathlib.Path, num, mode, start_index=0):
        if num == 0:
            return
        ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix.lower() in ['.png', '.jpg', '.bmp', '.tif', '.tiff']]
        vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix.lower() in ['.png', '.jpg', '.bmp', '.tif', '.tiff']]
        assert len(ir_list) > 0 and len(vi_list) > 0, f'empty folder: ir={ir_folder}, vi={vi_folder}'
        if [p.stem for p in ir_list] != [p.stem for p in vi_list]:
            raise ValueError('Aligned IR/VIS sources must have matching sorted filename stems')
        os.makedirs(dst / mode, exist_ok=True)
        os.makedirs(dst / mode / 'elastic_disp', exist_ok=True)
        os.makedirs(dst / mode / 'homo_shift', exist_ok=True)
        for cout in range(start_index, start_index + num):
            if cout % 100 == 0:
                print(f'{mode}: generating {cout + 1:06d}', flush=True)
            name_save = str(cout + 1).zfill(6) + '.png'
            name_disp = str(cout + 1).zfill(6) + '.npy'
            index = rd.randint(0, len(vi_list) - 1)
            ir_path = ir_list[index]
            vi_path = vi_list[index]
            ir = self.imread(ir_path, mini_wh=412, short_side=self.source_short_side).to(self.device)
            vi = self.imread(vi_path, mini_wh=412, short_side=self.source_short_side).to(self.device)
            (ir1, ir2, vis1, vis2, warp_grid, H, shift, elastic_disp) = self.homo_elastic(ir, vi, need_mask=False)
            np.save(dst / mode / 'elastic_disp' / name_disp, elastic_disp.data.cpu().numpy()[0])
            np.save(dst / mode / 'homo_shift' / name_disp, shift.detach().cpu().numpy()[0])
            self.imsave(ir1, dst / mode / 'ir_input1', name_save)
            self.imsave(ir2, dst / mode / 'ir_input2', name_save)
            self.imsave(vis1, dst / mode / 'vis_input1', name_save)
            self.imsave(vis2, dst / mode / 'vis_input2', name_save)
            self.imsave(warp_grid, dst / mode / 'warp_grid', name_save)

    @staticmethod
    def imread(path: pathlib.Path, mini_wh=None, unsqueeze=False, short_side=None):
        im_cv = cv2.imread(str(path))
        assert im_cv is not None, f'Image {str(path)} is invalid.'
        (h, w, _) = im_cv.shape
        if short_side is not None:
            scale = short_side / min(h, w)
            im_cv = cv2.resize(im_cv, (round(w * scale), round(h * scale)))
        elif mini_wh is not None:
            if h < mini_wh:
                scale = mini_wh / h
                (h, w) = (mini_wh, int(scale * w))
            if w < mini_wh:
                scale = mini_wh / w
                (w, h) = (mini_wh, int(scale * h))
            im_cv = cv2.resize(im_cv, (w, h))
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.0).type(torch.FloatTensor)
        return im_ts[None]

    @staticmethod
    def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str=''):
        im_s = im_s if type(im_s) == list else [im_s]
        dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
        for (im_ts, p) in zip(im_s, dst):
            im_ts = im_ts.squeeze().cpu()
            p.parent.mkdir(parents=True, exist_ok=True)
            im_cv = kornia.utils.tensor_to_image(im_ts) * 255.0
            cv2.imwrite(str(p), im_cv)

def make_generator(args):

    def corner_ranges(values):
        return None if values is None else np.asarray(values).reshape(4, 2)
    return getDeformableImages(shift_scale=args.shift_scale, trans=args.trans, shift=args.shift, patch_size=args.patch_size, corner_x_ranges=corner_ranges(args.corner_x_ranges), corner_y_ranges=corner_ranges(args.corner_y_ranges), random_crop=args.random_crop, crop_origin=args.crop_origin, source_short_side=args.source_short_side, device=args.device)

def generate_chunk(args, mode, start_index, count):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if args.seed is not None:
        worker_seed = (args.seed + start_index + (args.num_train if mode == 'test' else 0)) % 2 ** 32
        rd.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    make_generator(args)(args.ir, args.vi, args.dst, count, mode, start_index=start_index)
    return (mode, start_index, count)

def hyper_args():
    parser = argparse.ArgumentParser(description='Generating deformable registration bigfield data')
    parser.add_argument('--ir', type=pathlib.Path, default=DEFAULTS['ir'])
    parser.add_argument('--vi', type=pathlib.Path, default=DEFAULTS['vi'])
    parser.add_argument('--dst', type=pathlib.Path, default=DEFAULTS['dst'])
    parser.add_argument('--num_train', type=int, default=DEFAULTS['num_train'])
    parser.add_argument('--num_test', type=int, default=DEFAULTS['num_test'])
    parser.add_argument('--shift_scale', type=float, default=DEFAULTS['shift_scale'])
    parser.add_argument('--trans', type=float, help='Shared translation bound as a fraction of patch size', default=DEFAULTS['trans'])
    parser.add_argument('--shift', type=float, help='Symmetric corner bound when explicit corner ranges are omitted', default=DEFAULTS['shift'])
    parser.add_argument('--patch_size', type=int, default=DEFAULTS['patch_size'])
    parser.add_argument('--corner_x_ranges', nargs=8, type=float, help='xmin xmax for TL TR BL BR, as fractions of patch size', default=DEFAULTS['corner_x_ranges'])
    parser.add_argument('--corner_y_ranges', nargs=8, type=float, help='ymin ymax for TL TR BL BR, as fractions of patch size', default=DEFAULTS['corner_y_ranges'])
    parser.add_argument('--random_crop', action='store_true', help='Sample a valid crop origin instead of using the fixed origin', default=DEFAULTS['random_crop'])
    parser.add_argument('--crop_origin', nargs=2, type=int, metavar=('X', 'Y'), default=DEFAULTS['crop_origin'])
    parser.add_argument('--source_short_side', type=int, help='Resize source short side to this size; default only enlarges small sources to 412', default=DEFAULTS['source_short_side'])
    parser.add_argument('--seed', type=int, help='Seed Python, NumPy and PyTorch random generators', default=DEFAULTS['seed'])
    parser.add_argument('--workers', type=int, help='Independent generation processes; use 1 for original sequential RNG order', default=DEFAULTS['workers'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default=DEFAULTS['device'])
    parser.add_argument('--fixed_crop', action='store_false', dest='random_crop')
    args = parser.parse_args()
    if args.num_train < 0 or args.num_test < 0:
        parser.error('num_train and num_test must be nonnegative')
    if args.source_short_side is not None and args.source_short_side <= 0:
        parser.error('source_short_side must be positive')
    if args.workers < 1:
        parser.error('workers must be positive')
    return args
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = hyper_args()
    if args.seed is not None:
        rd.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    print(f'ir={args.ir}\nvi={args.vi}\ndst={args.dst}\nshift_scale={args.shift_scale}')
    data = make_generator(args)
    args.dst.mkdir(parents=True, exist_ok=True)
    with open(args.dst / 'generation_config.json', 'w', encoding='utf-8') as config_file:
        json.dump(vars(args), config_file, ensure_ascii=False, indent=2, default=str)
    if args.workers == 1:
        data(ir_folder=args.ir, vi_folder=args.vi, dst=args.dst, num=args.num_train, mode='train')
        data(ir_folder=args.ir, vi_folder=args.vi, dst=args.dst, num=args.num_test, mode='test')
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = []
            for (mode, total) in [('train', args.num_train), ('test', args.num_test)]:
                if total == 0:
                    continue
                chunk_size = (total + args.workers - 1) // args.workers
                for offset in range(0, total, chunk_size):
                    futures.append(pool.submit(generate_chunk, args, mode, offset, min(chunk_size, total - offset)))
            for future in futures:
                print('completed chunk:', future.result(), flush=True)
    print('Generation complete.', flush=True)
