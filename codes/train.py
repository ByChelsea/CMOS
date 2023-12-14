import os
import math
import argparse
import random
import logging
import numpy as np
import torch
import cv2
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from data.util import bgr2ycbcr

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

import socket
import getpass
from torchvision.transforms import ToPILImage


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':  # Return the name of start method used for starting processes
        mp.set_start_method('spawn', force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ['RANK'])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)  # Initializes the default distributed process group


def main():
    #### setup options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/train/train_stage1.yml',
                        help='Path to the option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu_ids_qsub', type=str, default=None)
    parser.add_argument('--slurm_job_id', type=str, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, args.gpu_ids_qsub, is_train=True)
    device_id = torch.cuda.current_device()

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    torch.backends.cudnn.benchmark = True

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info('{}@{}, GPU {}, Job_id {}, Job path {}'.format(getpass.getuser(), socket.gethostname(),
                                                                   opt['gpu_ids'], args.slurm_job_id, os.getcwd()))
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            if opt['train']['niter']:
                total_iters = int(opt['train']['niter'])
                total_epochs = int(math.ceil(total_iters / train_size))
            else:
                total_epochs = int(opt['train']['nepoch'])
                total_iters = total_epochs * train_size
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    # mixed precision
    scaler = torch.cuda.amp.GradScaler()

    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 10):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            GT_img, LR_img, blur_map, seg_map = \
                train_data['GT'], train_data['LR'], train_data['blur_map'], train_data['seg_map']

            #### training
            model.feed_data(GT_img, LR_img, blur_map, seg_map)
            model.optimize_parameters(current_step, scaler)

            #### update learning rate, schedulers
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}:{:.4e} '.format(k, v)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
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
                for _, val_data in enumerate(val_loader):
                    idx += 1

                    GT_img, LR_img, blur_map, seg_map = \
                        val_data['GT'], val_data['LR'], val_data['blur_map'], val_data['seg_map']

                    model.feed_data(GT_img, LR_img, blur_map, seg_map)
                    model.test()

                    visuals = model.get_current_visuals()

                    # Save SR images for reference
                    img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], str(current_step))
                    util.mkdir(img_dir)

                    # deal with blur
                    if opt['val_blur']:
                        save_blur_path = os.path.join(img_dir, 'blur')
                        util.mkdir(save_blur_path)
                        save_blur_path = os.path.join(save_blur_path, '{:s}_{:d}.png'.format(img_name, current_step))
                        blur_map_est = visuals['blur_map_est'][0] / ((opt['kernel_size'] - 1) / 4.0)
                        blur_map = visuals['blur_map'][0] / ((opt['kernel_size'] - 1) / 4.0)
                        cv2.imwrite(save_blur_path, np.vstack((blur_map_est, blur_map)) * 255.)
                        avg_psnr_b += util.calculate_psnr(blur_map_est, blur_map, 1.)
                        avg_ssim_b += util.calculate_ssim(blur_map_est * 255., blur_map * 255.)

                    # deal with seg
                    if opt['val_seg']:
                        save_seg_path = os.path.join(img_dir, 'seg')
                        util.mkdir(save_seg_path)
                        save_seg_path = os.path.join(save_seg_path, '{:s}_{:d}.png'.format(img_name, current_step))
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
                        sr_img = util.tensor2img(visuals['SR'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        gt_img = gt_img / 255.
                        sr_img = sr_img / 255.
                        crop_size = opt['crop_border'] if opt['crop_border'] else opt['scale']
                        if gt_img.shape[2] == 3:  # RGB image
                            sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                            gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                            if crop_size == 0:
                                cropped_sr_img_y = sr_img_y
                                cropped_gt_img_y = gt_img_y
                            else:
                                cropped_sr_img_y = sr_img_y[crop_size:-crop_size, crop_size:-crop_size]
                                cropped_gt_img_y = gt_img_y[crop_size:-crop_size, crop_size:-crop_size]
                            avg_psnr_y += util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255, 255.)
                            avg_ssim_y += util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)

                avg_psnr_y = avg_psnr_y / idx
                avg_ssim_y = avg_ssim_y / idx
                avg_psnr_b = avg_psnr_b / idx
                avg_ssim_b = avg_ssim_b / idx
                jac = [0] * n_classes
                for i_part in range(0, n_classes):
                    jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)
                avg_miou = np.mean(jac)

                # log
                logger.info('# {}, Validation # PSNR_Y: {:.4f},  SSIM_Y: {:.4f},  PSNR_B: {:.4f}, SSIM_B: {:.4f}, '
                            'MIOU: {:.4f}'.format(opt['name'], avg_psnr_y, avg_ssim_y, avg_psnr_b, avg_ssim_b, avg_miou))
                logger.info('{}@{}, GPU {}'.format(getpass.getuser(), socket.gethostname(), opt['gpu_ids']))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr_y: {:.6f}, ssim_y: {:.6f}, psnr_b: {:.6f}, '
                                'ssim_b: {:.6f}, miou: {:.6f}'.format(epoch, current_step, avg_psnr_y, avg_ssim_y,
                                                                      avg_psnr_b, avg_ssim_b, avg_miou))

                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr_y', avg_psnr_y, current_step)
                    tb_logger.add_scalar('ssim_y', avg_ssim_y, current_step)
                    tb_logger.add_scalar('psnr_b', avg_psnr_b, current_step)
                    tb_logger.add_scalar('ssim_b', avg_ssim_b, current_step)
                    tb_logger.add_scalar('miou', avg_miou, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()