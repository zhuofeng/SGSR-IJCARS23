# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from mmcv.runner import load_checkpoint
from torch.nn import functional as F
import pdb
import os
import nibabel as nib
import numpy as np

from mmedit.utils import get_root_logger
from ..registry import LOSSES
from .common import default_conv, ResBlock, Upsampler


@LOSSES.register_module()
class UDLmodel(nn.Module):
    def __init__(self):
        super(UDLmodel, self).__init__()
        conv=default_conv
        n_resblock = 16
        n_feats = 64
        kernel_size = 3
        scale = 4
        res_scale = 0.1
        n_colors = 3
        # act = nn.ReLU(True)
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_factor = 4

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.var_conv = nn.Sequential(*[conv(n_feats, n_feats, kernel_size), nn.ELU(),conv(n_feats, n_feats, kernel_size), nn.ELU(),conv(n_feats, n_colors, kernel_size), nn.ELU()])
        
        # self.var_conv = nn.Sequential(*[conv(n_feats, n_feats, kernel_size), nn.LeakyReLU(negative_slope=0.2, inplace=True), conv(n_feats, n_feats, kernel_size), nn.LeakyReLU(negative_slope=0.2, inplace=True), conv(n_feats, args.n_colors, kernel_size), nn.LeakyReLU(negative_slope=0.2, inplace=True)])

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        var = self.var_conv(res)
        return [x, var]


@LOSSES.register_module()
class UDLLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'4': 1., '9': 1., '18': 1.}, which means the
            5th, 10th and 18th feature layer will be extracted with weight 1.0
            in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def __init__(self,
                 loss_weight=1.0,
                 pretrained='torchvision://vgg19',
                 criterion='l1'):
        super().__init__()
        self.loss_weight = loss_weight
        self.UDLmodel = UDLmodel()
        self.init_weights(self.UDLmodel, pretrained)
        for v in self.UDLmodel.parameters():
            v.requires_grad = False

        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')
        
    def init_weights(self, model, pretrained):
        """Init weights.

        Args:
            model (nn.Module): Models to be inited.
            pretrained (str): Path for pretrained weights.
        """
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)
        
    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        
        # additional operation (make x and gt blurred to calculate uncertainty. This makes formular more precise)
        x = F.interpolate(x,  scale_factor = 0.25, mode='bicubic', align_corners=False)
        x = F.interpolate(x, scale_factor = 4, mode='bicubic', align_corners=False)
        gt = F.interpolate(gt, scale_factor = 0.25, mode='bicubic', align_corners=False)
        gt = F.interpolate(gt, scale_factor = 4, mode='bicubic', align_corners=False)
        
        # extract vgg features
        if x.shape[1] != 3:
            x_input = x.repeat(1,3,1,1)
            gt_input = gt.repeat(1,3,1,1) 
            x_SR = self.UDLmodel(x_input)[1]
            gt_SR = self.UDLmodel(gt_input)[1]
        else:
            x_SR = self.UDLmodel(x)[1]
            gt_SR = self.UDLmodel(gt)[1]
        
        b, c, h, w = x_SR.shape
        x_SR_1 = x_SR.view(b, c, -1)
        x_SR_min = torch.min(x_SR_1, dim=-1)
        x_SR_min = x_SR_min[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        x_SR = x_SR - x_SR_min+1

        b, c, h, w = gt_SR.shape
        gt_SR_1 = gt_SR.view(b, c, -1)
        gt_SR_min = torch.min(gt_SR_1, dim=-1)
        gt_SR_min = gt_SR_min[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        gt_SR = gt_SR - gt_SR_min+1

        commonuncer = x_SR * gt_SR # the common uncertainty x_SR and gt_SR
        
        # before calculating it, normalize x and gt
        
        x = x - x.min()
        gt = gt - gt.min()
        udl_loss = self.criterion(
                    x * commonuncer, gt * commonuncer) * self.loss_weight
        
        '''
        # save to have a look
        x = x[:,1,:,:].cpu().detach().numpy().squeeze()
        gt = gt[:,1,:,:].cpu().detach().numpy().squeeze()
        
        
        x_SR = x_SR[1][:,1,:,:].cpu().detach().numpy().squeeze()
        gt_SR = gt_SR[1][:,1,:,:].cpu().detach().numpy().squeeze()
        commonuncer = commonuncer[:,1,:,:].cpu().detach().numpy().squeeze()
        
        x_mul_uncer = x*commonuncer
        gt_mul_uncer = gt*commonuncer
        
        x_mul_uncer = nib.Nifti1Image(x_mul_uncer, np.eye(4))
        nib.save(x_mul_uncer, '/homes/tzheng/code/mmediting/outputs/x_mul_uncer.nii.gz')
        gt_mul_uncer = nib.Nifti1Image(gt_mul_uncer, np.eye(4))
        nib.save(gt_mul_uncer, '/homes/tzheng/code/mmediting/outputs/gt_mul_uncer.nii.gz')
        x_SR = nib.Nifti1Image(x_SR, np.eye(4))
        nib.save(x_SR, '/homes/tzheng/code/mmediting/outputs/x_SR.nii.gz')
        gt_SR = nib.Nifti1Image(gt_SR, np.eye(4))
        nib.save(gt_SR, '/homes/tzheng/code/mmediting/outputs/gt_SR.nii.gz')
        commonuncer = nib.Nifti1Image(commonuncer, np.eye(4))
        nib.save(commonuncer, '/homes/tzheng/code/mmediting/outputs/commonuncer.nii.gz')
        x = nib.Nifti1Image(x, np.eye(4))
        gt = nib.Nifti1Image(gt, np.eye(4))
        nib.save(x, '/homes/tzheng/code/mmediting/outputs/x.nii.gz')
        nib.save(gt, '/homes/tzheng/code/mmediting/outputs/gt.nii.gz')
        '''
        return udl_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
