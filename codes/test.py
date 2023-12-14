import os.path
import logging
import argparse
from collections import OrderedDict
import numpy as np
import torch
import cv2
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from torchvision.transforms import ToPILImage

#### options
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='options/test/NYUv2_BSR/test_stage3.yml', help='Path to options YMAL file.')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)
device_id = torch.cuda.current_device()

#### mkdir and logger
util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
             and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

# set random seed
util.set_random_seed(0)

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']  # path opt['']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))

    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    n_classes = opt['n_classes']
    avg_psnr_y = 0.0
    avg_ssim_y = 0.0
    avg_psnr_b = 0.0
    avg_ssim_b = 0.0
    avg_miou = 0.0
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    idx = 0
    for test_data in test_loader:
        idx += 1

        GT_img, LR_img, blur_map, seg_map = \
            test_data['GT'], test_data['LR'], test_data['blur_map'], test_data['seg_map']

        model.feed_data(GT_img, LR_img, blur_map, seg_map)
        model.test()

        visuals = model.get_current_visuals()

        # Save SR images for reference
        img_name = os.path.splitext(os.path.basename(test_data['LR_path'][0]))[0]

        # deal with blur
        if opt['val_blur']:
            save_blur_path = os.path.join(dataset_dir, 'blur')
            util.mkdir(save_blur_path)
            save_blur_path = os.path.join(save_blur_path, '{:s}.png'.format(img_name))
            blur_map_est = visuals['blur_map_est'][0] / ((opt['kernel_size'] - 1) / 4.0)
            blur_map = visuals['blur_map'][0] / ((opt['kernel_size'] - 1) / 4.0)
            cv2.imwrite(save_blur_path, np.vstack((blur_map_est, blur_map)) * 255.)
            avg_psnr_b += util.calculate_psnr(blur_map_est, blur_map, 1.)
            avg_ssim_b += util.calculate_ssim(blur_map_est * 255., blur_map * 255.)

        # deal with seg
        if opt['val_seg']:
            save_seg_path = os.path.join(dataset_dir, 'seg')
            util.mkdir(save_seg_path)
            save_seg_path = os.path.join(save_seg_path, '{:s}.png'.format(img_name))
            _, seg_map_est = torch.max(visuals['seg_map_est'], dim=0)
            seg_map = visuals['seg_map']
            seg_map_est = np.array(seg_map_est).astype(np.uint8).astype(np.float32)  # visualization
            label_color = util.Colorize()(seg_map_est)
            label_save = ToPILImage()(label_color)
            label_save.save(save_seg_path)
            valid = (seg_map != 255)
            for i_part in range(0, n_classes):
                tmp_gt = (seg_map == i_part)
                tmp_pred = (seg_map_est == i_part)
                tp[i_part] += np.sum(tmp_gt & tmp_pred & valid)
                fp[i_part] += np.sum(~tmp_gt & tmp_pred & valid)
                fn[i_part] += np.sum(tmp_gt & ~tmp_pred & valid)

        # deal with sr image
        if opt['val_sr']:
            sr_img = util.tensor2img(visuals['SR']) 
            gt_img = util.tensor2img(visuals['GT'])
            save_img_path = os.path.join(dataset_dir, '{:s}.png'.format(img_name))
            util.save_img(sr_img, save_img_path)

            # calculate PSNR
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.
            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                avg_psnr_y += util.calculate_psnr(sr_img_y * 255, gt_img_y * 255, 255.)
                avg_ssim_y += util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)

    avg_psnr_y = avg_psnr_y / idx
    avg_ssim_y = avg_ssim_y / idx
    avg_psnr_b = avg_psnr_b / idx
    avg_ssim_b = avg_ssim_b / idx
    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)
    avg_miou = np.mean(jac)

    # log
    logger.info('# {}, Test # PSNR_Y: {:.4f},  SSIM_Y: {:.4f},  PSNR_B: {:.4f}, SSIM_B: {:.4f}, '
                'MIOU: {:.4f}'.format(opt['name'], avg_psnr_y, avg_ssim_y, avg_psnr_b, avg_ssim_b, avg_miou))
