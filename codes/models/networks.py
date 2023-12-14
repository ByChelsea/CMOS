import torch
import logging
import models.modules.CMOS_arch as CMOS_arch

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'CMOS':
        netG = CMOS_arch.CMOSNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                 nb=opt_net['nb'], gc=opt_net['gc'], scale=opt['scale'], code_length=opt['code_length'],
                                 seg_classes=opt['n_classes'])
    elif which_model == 'RRDB_SFT':
        netG = CMOS_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                 nb=opt_net['nb'], gc=opt_net['gc'], scale=opt['scale'],  code_length=opt['code_length'],
                                 seg_classes=opt['n_classes'])
    elif which_model == 'BlindSR':
        netG = CMOS_arch.BlindSR(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                 nb=opt_net['nb'], gc=opt_net['gc'], scale=opt['scale'], code_length=opt['code_length'],
                                 seg_classes=opt['n_classes'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG
