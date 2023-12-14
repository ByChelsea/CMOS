import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from collections import OrderedDict
import models.modules.module_util as mutil
from models.modules.seg_hrnet import hrnet_w18, HighResolutionHead
import torch.nn.init as init
from models.modules.resnet import BasicBlock
from models.modules.layers import SEBlock
from mmcv.cnn import ConvModule
import torchvision


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# -------------------------------------------------
# SFT layer and RRDB block for non-blind RRDB-SFT
# -------------------------------------------------

class SFT_Layer(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB_SFT(nn.Module):
    def __init__(self, nf, gc=32, para=15):
        super(RRDB_SFT, self).__init__()
        self.SFT = SFT_Layer(nf=nf, para=para)
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, input):
        out = self.SFT(input[0], input[1])
        out = self.RDB1(out)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return [out * 0.2 + input[0], input[1]]


# --------------------------------
# CMOS architecture and GIA module
# --------------------------------

class InitialTaskPredictionModule(nn.Module):
    def __init__(self, tasks, channels, out_channels, scale):
        super(InitialTaskPredictionModule, self).__init__()
        self.tasks = tasks

        self.refinement = nn.ModuleDict(
            {task: nn.Sequential(BasicBlock(channels, channels), BasicBlock(channels, channels)) for task in
             self.tasks})

        if scale == 0:
            self.decoders = nn.ModuleDict(
                {task: nn.Conv2d(channels, out_channels[task], 1) for task in self.tasks})

    def forward(self, scale, features_curr_scale):
        if scale == 3:
            x = {t: features_curr_scale for t in self.tasks}
        else:
            x = features_curr_scale

        out = {}
        for t in self.tasks:
            out['features_%s' % t] = self.refinement[t](x[t])
            if scale == 0:
                out[t] = self.decoders[t](out['features_%s' % t])

        return out


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.permute(0, 2, 3, 1).contiguous()
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    return x


class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super(ChannelAtt, self).__init__()
        self.conv_bn_relu = ConvModule(in_channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x, fre=False):
        feat = self.conv_bn_relu(x)
        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, atten


class GIAChannel(nn.Module):
    def __init__(self, channels, conv_cfg, norm_cfg, act_cfg, r):
        super(GIAChannel, self).__init__()
        self.r = r
        self.c1, self.c2 = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        self.channel_att1, self.channel_att2 = [ChannelAtt(channels, channels, conv_cfg, norm_cfg, act_cfg) for _ in
                                                range(2)]
        self.mlp1, self.mlp2 = [nn.Sequential(
            nn.Linear(r * r, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        ) for _ in range(2)]
        self.smooth1, self.smooth2 = [
            ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(2)]

    def forward(self, x1, x2):
        feat1, att1 = self.channel_att1(x1)
        feat2, att2 = self.channel_att2(x2)
        b, c, h, w = att1.size()
        att1_split = att1.view(b, self.r, c // self.r)
        att2_split = att2.view(b, self.r, c // self.r)
        chl_affinity = torch.bmm(att2_split, att1_split.permute(0, 2, 1))
        chl_affinity = chl_affinity.view(b, -1)
        re_att1 = F.relu(self.mlp1(chl_affinity))
        re_att2 = F.relu(self.mlp2(chl_affinity))
        re_att1 = torch.sigmoid(att1 + self.c1 * re_att1.unsqueeze(-1).unsqueeze(-1))
        re_att2 = torch.sigmoid(att2 + self.c2 * re_att2.unsqueeze(-1).unsqueeze(-1))
        feat1 = self.smooth1(torch.mul(feat1, re_att1))
        feat2 = self.smooth2(torch.mul(feat2, re_att2))
        return feat1, feat2


class GIASpatial(nn.Module):
    def __init__(self, channels, conv_cfg, norm_cfg, act_cfg, win_size=(15, 20)):
        super(GIASpatial, self).__init__()
        self.win_size = win_size
        self.s1, self.s2 = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        self.conv_first1, self.conv_first2 = [
            ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                       act_cfg=act_cfg) for _ in range(2)]
        self.conv1, self.conv2 = [nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0) for _ in range(2)]
        self.conv_after_window1, self.conv_after_window2 = [
            ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                       act_cfg=act_cfg) for _ in range(2)]
        self.conv_redu1, self.conv_redu2 = [nn.Conv2d(win_size[0] * win_size[1], 1, kernel_size=1, stride=1,
                                                      padding=0) for _ in range(2)]
        self.smooth1, self.smooth2 = [
            ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(2)]

    def forward(self, x1, x2):
        feat1 = self.conv_first1(x1)
        feat2 = self.conv_first2(x2)
        # padding
        b_ori, c_ori, h_ori, w_ori = feat1.size()
        mod_pad_h = (self.win_size[0] - h_ori % self.win_size[0]) % self.win_size[0]
        mod_pad_w = (self.win_size[1] - w_ori % self.win_size[1]) % self.win_size[1]
        feat1 = F.pad(feat1, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        feat2 = F.pad(feat2, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # to windows
        B, C, H, W = feat1.size()
        windows1 = window_partition(feat1, self.win_size).permute(0, 3, 1, 2).contiguous()
        windows2 = window_partition(feat2, self.win_size).permute(0, 3, 1, 2).contiguous()
        windows_conv1 = self.conv_after_window1(windows1)
        windows_conv2 = self.conv_after_window2(windows2)
        b, c, h, w = windows_conv1.size()
        windows_conv1 = windows_conv1.view(b, c, -1)
        windows_conv2 = windows_conv2.view(b, c, -1)
        spa_affinity = torch.bmm(windows_conv1.permute(0, 2, 1), windows_conv2)
        re_windows1 = spa_affinity.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        re_windows2 = spa_affinity.view(b, -1, h, w)
        re_windows1 = torch.sigmoid(self.conv1(windows1) + self.s1 * self.conv_redu1(re_windows1))
        re_windows2 = torch.sigmoid(self.conv2(windows2) + self.s2 * self.conv_redu2(re_windows2))
        feat1 = torch.mul(windows1, re_windows1)
        feat2 = torch.mul(windows2, re_windows2)
        # back to features
        feat1 = window_reverse(feat1, self.win_size, H, W)
        feat2 = window_reverse(feat2, self.win_size, H, W)
        feat1 = self.smooth1(feat1[:, :, :h_ori, :w_ori])
        feat2 = self.smooth2(feat2[:, :, :h_ori, :w_ori])
        return feat1, feat2


class AlignedModule(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(outplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)
        self.conv_last = nn.Conv2d(inplane, outplane, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, h_feature, low_feature, upsample):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        if upsample:
            h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        h_feature = self.conv_last(h_feature)
        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class GIA(nn.Module):
    def __init__(self, channels_l, channels_h, conv_cfg, norm_cfg, act_cfg, r, win_size, upsample):
        super(GIA, self).__init__()
        self.upsample = upsample

        self.upsample_layer = AlignedModule(channels_l, channels_h)
        self.channel = GIAChannel(channels_h, conv_cfg, norm_cfg, act_cfg, r)
        self.spatial = GIASpatial(channels_h, conv_cfg, norm_cfg, act_cfg, win_size)

        self.smooth1, self.smooth2 = [
            ConvModule(channels_h * 2, channels_h, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                       act_cfg=None) for _ in range(2)]

    def forward(self, x1, x2, num):
        x1 = self.upsample_layer(x1, x2, self.upsample)
        ch_feat1, ch_feat2 = self.channel(x1, x2)
        sp_feat1, sp_feat2 = self.spatial(x1, x2)
        final_x1 = self.smooth2(torch.cat([ch_feat1, sp_feat1], 1))
        final_x2 = self.smooth1(torch.cat([ch_feat2, sp_feat2], 1))

        if num == 2:
            return final_x1, final_x2
        else:
            return final_x1 + final_x2


class CMOS(nn.Module):
    def __init__(self, scale=4, backbone_channels=[18, 36, 72, 144], tasks=['seg', 'blur'],
                 r=[3, 6, 8, 12], win_size=(15, 20), seg_classes=40, pretrained=True):
        super(CMOS, self).__init__()
        self.scale = scale
        self.tasks = tasks
        self.num_scales = len(backbone_channels)
        self.channels = backbone_channels

        # stage 1
        self.backbone = hrnet_w18(pretrained)

        # stage 2
        out_channels = {'seg': seg_classes, 'blur': 1}
        self.heads = nn.ModuleList(
            [InitialTaskPredictionModule(self.tasks, self.channels[s], out_channels, s) for s in range(4)])
        self.GIA_m = nn.ModuleList(
            [GIA(self.channels[s], self.channels[s], conv_cfg=None, norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LeakyReLU'), r=r[s], win_size=win_size, upsample=False) for s in range(1, 4)])
        self.GIA_s = nn.ModuleList(
            [GIA(self.channels[s], self.channels[s - 1], conv_cfg=None, norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LeakyReLU'), r=r[s - 1], win_size=win_size, upsample=True) for s in range(1, 4)])
        self.GIA_b = nn.ModuleList(
            [GIA(self.channels[s], self.channels[s - 1], conv_cfg=None, norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LeakyReLU'), r=r[s - 1], win_size=win_size, upsample=True) for s in range(1, 4)])

        # stage 3
        self.GIA_l = nn.ModuleList(
            [GIA(self.channels[s], self.channels[s], conv_cfg=None, norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LeakyReLU'), r=r[s], win_size=win_size, upsample=False) for s in range(4)])
        self.predict = torch.nn.ModuleDict(
            {task: HighResolutionHead(backbone_channels, out_channels[task]) for task in self.tasks})

    def forward(self, x):
        # stage 1
        x = self.backbone(x)

        # stage 2
        x_scale, x_bs, x_hat_seg, x_hat_blur = {}, {}, {}, {}
        for i in range(3, 0, -1):  # 3,2,1
            x_scale[i] = self.heads[i](i, x_bs[i + 1]) if i != 3 else self.heads[i](i, x[i])  # task-specific heads
            x_hat_blur[i], x_hat_seg[i] = self.GIA_m[i - 1](x_scale[i]['features_blur'], x_scale[i]['features_seg'], 2)
            x_bs[i] = {}
            x_bs[i]['seg'] = self.GIA_s[i - 1](x_hat_seg[i], x[i - 1], 1)
            x_bs[i]['blur'] = self.GIA_b[i - 1](x_hat_blur[i], x[i - 1], 1)

        x_scale[0] = self.heads[0](0, x_bs[1])

        # stage 3
        features = {}
        for i in range(4):
            features[i] = {}
            features[i]['blur'], features[i]['seg'] = \
                self.GIA_l[i](x_scale[i]['features_blur'], x_scale[i]['features_seg'], 2)
        multi_scale_features = {t: [features[0][t], features[1][t], features[2][t], features[3][t]] for t in self.tasks}
        out = {}
        for t in self.tasks:
            out[t] = F.interpolate(self.predict[t](multi_scale_features[t]), scale_factor=self.scale, mode='bilinear')
            x_scale[0][t] = F.interpolate(x_scale[0][t], scale_factor=self.scale, mode='bilinear')

        return out['blur'], out['seg'], x_scale[0]


class CMOSNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, code_length=15, seg_classes=40):
        super(CMOSNet, self).__init__()
        self.scale = scale
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.cmos = CMOS(scale=self.scale, backbone_channels=[18, 36, 72, 144], tasks=['seg', 'blur'],
                         r=[3, 6, 8, 12], win_size=(15, 20), seg_classes=seg_classes, pretrained=True)

    def forward(self, x, gt_B, gt_S):
        # no meaning
        with torch.no_grad():
            out = F.interpolate(x, scale_factor=self.scale, mode='nearest')

        x = self.normalize(x)
        blur, seg, deep_sup = self.cmos(x)
        return out, blur, seg, deep_sup['blur'], deep_sup['seg']


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, code_length=15, seg_classes=40):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.seg_classes = seg_classes

        self.blur_conv = nn.Conv2d(1, code_length, 3, 1, 1, bias=True)
        self.seg_conv = nn.Conv2d(1, code_length, 3, 1, 1, bias=True)
        self.fuse = GIA(code_length, code_length, conv_cfg=None, norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='LeakyReLU'), r=int(code_length ** 0.5), win_size=(15, 20), upsample=False)
        RRDB_SFT_block_f = functools.partial(RRDB_SFT, nf=nf, gc=gc, para=code_length)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_SFT_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upsampler = sequential(nn.Conv2d(nf, out_nc * (scale ** 2), kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.PixelShuffle(scale))

    def forward(self, x, gt_B, gt_S):
        blur = self.blur_conv(gt_B)
        seg = self.seg_conv(gt_S)
        prior_information = self.fuse(blur, seg, 1)
        prior_information = F.interpolate(prior_information, scale_factor=1. / self.scale, mode='nearest')

        # nonblind sr
        lr_fea = self.conv_first(x)
        fea = self.RRDB_trunk([lr_fea, prior_information])
        fea = lr_fea + self.trunk_conv(fea[0])
        out = self.upsampler(fea)
        return out, gt_B, gt_S, gt_B, gt_S


class BlindSR(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, code_length=15, seg_classes=40):
        super(BlindSR, self).__init__()
        self.scale = scale
        # CMOS
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.cmos = CMOS(scale=self.scale, backbone_channels=[18, 36, 72, 144], tasks=['seg', 'blur'],
                         r=[3, 6, 8, 12], win_size=(15, 20), seg_classes=seg_classes, pretrained=True)
        # RRDB_SFT
        self.blur_conv = nn.Conv2d(1, code_length, 3, 1, 1, bias=True)
        self.seg_conv = nn.Conv2d(1, code_length, 3, 1, 1, bias=True)
        self.fuse = GIA(code_length, code_length, conv_cfg=None, norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='LeakyReLU'), r=int(code_length ** 0.5), win_size=(15, 20), upsample=False)
        RRDB_SFT_block_f = functools.partial(RRDB_SFT, nf=nf, gc=gc, para=64)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_SFT_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upsampler = sequential(nn.Conv2d(nf, out_nc * (scale ** 2), kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.PixelShuffle(scale))

    def forward(self, x, gt_B, gt_S):
        # cmos
        with torch.no_grad():
            blur_est, seg_est, _ = self.cmos(self.normalize(x))
            _, seg = torch.max(seg_est, dim=1, keepdim=True)

        blur = self.blur_conv(blur_est)
        seg = self.seg_conv(seg.float())
        prior_information = self.fuse(blur, seg, 1)
        prior_information = F.interpolate(prior_information, scale_factor=1. / self.scale, mode='nearest')

        # nonblind sr
        lr_fea = self.conv_first(x)
        fea = self.RRDB_trunk([lr_fea, prior_information])
        fea = lr_fea + self.trunk_conv(fea[0])
        out = self.upsampler(fea)
        return out, blur_est, seg_est, blur_est, seg_est


if __name__ == '__main__':
    model = CMOS()
    print(model)

    x = torch.randn((2, 3, 100, 100))
    blur, seg, _ = model(x)
    print(blur.shape, seg.shape)
