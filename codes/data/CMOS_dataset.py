import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import sys
import os
import torchvision
from PIL import Image

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
    from utils import util as utils
except ImportError:
    pass


class CMOSDataset(data.Dataset):
    def __init__(self, opt):
        super(CMOSDataset, self).__init__()
        self.opt = opt
        self.GT_env, self.LR_env, self.blur_env, self.seg_env = None, None, None, None  # environment for lmdb
        self.GT_size, self.LR_size = opt['GT_size'], opt['LR_size']

        # read image list from image files
        self.GT_paths = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])
        self.LR_paths = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        self.blur_paths = util.get_image_paths(opt['data_type'], opt['dataroot_blur'])
        self.seg_paths = util.get_image_paths(opt['data_type'], opt['dataroot_seg'])

        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        GT_path = self.GT_paths[index]
        LR_path = self.LR_paths[index]
        blur_path = self.blur_paths[index]
        if self.seg_paths is not None:
            seg_path = self.seg_paths[index]
        else:
            seg_path = None

        img_GT = np.array(Image.open(GT_path)).astype(np.float32)
        img_LR = np.array(Image.open(LR_path)).astype(np.float32)
        blur_map = (np.float32(cv2.imread(blur_path, cv2.IMREAD_UNCHANGED))/10.)[:, :, 1]
        if seg_path is not None:
            if 'NYUv2' in self.opt['name']:
                seg_map = np.array(Image.open(seg_path)).astype(np.float32)
                seg_map[seg_map == 0] = 256
                seg_map = seg_map - 1
            else:
                seg_map = np.array(Image.open(seg_path)).astype(np.float32)[:, :, 0]
        else:
            seg_map = np.zeros_like(blur_map)

        if self.opt['phase'] == 'train':
            H, W, C = img_LR.shape
            # randomly crop
            rnd_h_LR = random.randint(0, max(0, H - self.LR_size[0]))
            rnd_w_LR = random.randint(0, max(0, W - self.LR_size[1]))
            img_LR = img_LR[rnd_h_LR:rnd_h_LR + self.LR_size[0], rnd_w_LR:rnd_w_LR + self.LR_size[1], :]
            rnd_h = rnd_h_LR * self.opt['scale']
            rnd_w = rnd_w_LR * self.opt['scale']
            blur_map = blur_map[rnd_h:rnd_h + self.GT_size[0], rnd_w:rnd_w + self.GT_size[1]]
            seg_map = seg_map[rnd_h:rnd_h + self.GT_size[0], rnd_w:rnd_w + self.GT_size[1]]
            img_GT = img_GT[rnd_h:rnd_h + self.GT_size[0], rnd_w:rnd_w + self.GT_size[1], :]

            # augmentation - flip, rotate
            img_dict = {'GT': img_GT, 'LR': img_LR, 'blur': np.expand_dims(blur_map, axis=2),
                        'seg': np.expand_dims(seg_map, axis=2)}
            img_dict = util.augment(img_dict, self.opt['use_flip'], self.opt['use_rot'], self.opt['use_scale'])
            img_GT, img_LR, blur_map, seg_map = img_dict['GT'], img_dict['LR'], img_dict['blur'], img_dict['seg']

        # numpy to tensor
        img_GT = self.to_tensor(img_GT.astype(np.uint8))
        img_LR = self.to_tensor(img_LR.astype(np.uint8))
        blur_map = torch.from_numpy(np.ascontiguousarray(np.expand_dims(blur_map, axis=0))).squeeze(-1).float()
        seg_map = torch.from_numpy(np.ascontiguousarray(np.expand_dims(seg_map, axis=0))).squeeze(-1).float()

        return {'GT': img_GT, 'LR': img_LR, 'blur_map': blur_map, 'seg_map': seg_map,
                'GT_path': GT_path, 'LR_path': LR_path}

    def __len__(self):
        return len(self.LR_paths)

