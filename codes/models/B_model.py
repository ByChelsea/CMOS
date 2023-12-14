# base model for blind SR, input LR, output kernel + SR
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.modules.module import Module
from torch.cuda.amp import autocast as autocast
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
import utils.util as util

logger = logging.getLogger('base')


class B_Model(BaseModel):
    def __init__(self, opt):
        super(B_Model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()
        self.load_K()  # load the kernel estimation part

        if self.is_train:
            train_opt = opt['train']
            self.netG.train()

            # HR loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type is None:
                self.cri_pix = None
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # cmos loss
            loss_type = train_opt['cmos_criterion']
            if loss_type == 'cmos':
                self.cri_blur = nn.L1Loss().to(self.device)
                self.cri_seg = SoftMaxwithLoss().to(self.device)
            elif loss_type is None:
                self.cri_blur = None
                self.cri_seg = None
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_cmos_w = train_opt['cmos_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                print('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def init_model(self, scale=0.1):
        # Common practise for initialization.
        for layer in self.netG.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale  # for residual block
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)

    def feed_data(self, GT_img, LR_img, blur_map, seg_map):
        self.real_H, self.LR_img, self.real_B, self.real_S = \
            GT_img.to(self.device), LR_img.to(self.device), blur_map.to(self.device), seg_map.to(self.device)

    def optimize_parameters(self, step, scaler):
        self.optimizer_G.zero_grad()

        with autocast():
            l_all = 0
            self.fake_SR, self.fake_B, self.fake_S, self.sup_B, self.sup_S = self.netG(self.LR_img, self.real_B, self.real_S)

            # hr loss
            if self.cri_pix is not None:
                l_pix = self.l_pix_w * self.cri_pix(self.fake_SR, self.real_H)
                l_all += l_pix
                self.log_dict['l_pix'] = l_pix.item()

            # cmos loss
            if self.cri_blur is not None:
                l_blur_deep = self.l_cmos_w[0] * self.cri_blur(self.sup_B, self.real_B)
                l_seg_deep = self.l_cmos_w[1] * self.cri_seg(self.sup_S, self.real_S)
                l_blur = self.l_cmos_w[2] * self.cri_blur(self.fake_B, self.real_B)
                l_seg = self.l_cmos_w[3] * self.cri_seg(self.fake_S, self.real_S)
                l_cmos = l_blur_deep + l_seg_deep + l_blur + l_seg
                l_all += l_cmos
                self.log_dict['l_blur_deep'] = l_blur_deep.item()
                self.log_dict['l_seg_deep'] = l_seg_deep.item()
                self.log_dict['l_blur'] = l_blur.item()
                self.log_dict['l_seg'] = l_seg.item()
                self.log_dict['l_cmos'] = l_cmos.item()

        scaler.scale(l_all).backward()
        scaler.step(self.optimizer_G)
        scaler.update()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_SR, self.fake_B, self.fake_S, self.sup_B, self.sup_S = self.netG(self.LR_img, self.real_B, self.real_S)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR'] = self.LR_img.detach()[0].float().cpu().numpy()
        out_dict['SR'] = self.fake_SR.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['blur_map'] = self.real_B.detach()[0].float().cpu().numpy()
        out_dict['blur_map_est'] = self.fake_B.detach()[0].float().cpu().numpy()
        out_dict['seg_map'] = self.real_S.detach()[0].float().cpu().numpy()
        out_dict['seg_map_est'] = self.fake_S.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def load_K(self):
        load_path_K = self.opt['path']['pretrain_model_K']
        if load_path_K is not None:
            logger.info('Loading model for K [{:s}] ...'.format(load_path_K))
            self.load_network(load_path_K, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)


class SoftMaxwithLoss(Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=255)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        loss = self.criterion(self.softmax(out), label)

        return loss
