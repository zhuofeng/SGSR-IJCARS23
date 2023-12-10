# Copyright (c) OpenMMLab. All rights reserved.
import pdb
import os
import numpy as np
import logging

import torch.nn.functional as F
from mmcv.runner import auto_fp16
import torch
import nibabel as nib
from mmcv.cnn.utils import flops_counter

from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from ..registry import MODELS
from .basic_restorer import BasicRestorer

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8


@MODELS.register_module()
class SRGAN(BasicRestorer):
    """SRGAN model for single image super-resolution.

    Ref:
    Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network.

    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator. Default: None.
        gan_loss (dict): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict): Config for the pixel loss. Default: None.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 udl_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BasicRestorer, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(generator)
        
        # discriminator
        self.discriminator = build_component(
            discriminator) if discriminator else None

        # support fp16
        self.fp16_enabled = False

        # loss
        self.gan_loss = build_loss(gan_loss) if gan_loss else None
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None
        # self.perceptual_loss = build_loss(
        #     perceptual_loss) if perceptual_loss else None
        self.udl_loss = build_loss(
            udl_loss) if udl_loss else None
        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        self.step_counter = 0  # counting training steps

        self.gt_ema = None
        self.SR_ema = None
        self.lq_ema = None

        self.vxm_losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2', loss_mult=2).loss]
        self.vxm_weights = [0.01, 0.001, 0.01]
        self.uncerloss = torch.nn.L1Loss()
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained=pretrained)
        if self.discriminator:
            self.discriminator.init_weights(pretrained=pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, inference_example=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if inference_example:
            return self.forward_example(lq, gt, **kwargs)
        elif test_mode:
            return self.forward_test(lq, gt, **kwargs)

        raise ValueError(
            'SRGAN model does not support `forward_train` function.')

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # data_batch.keys(): dict_keys(['meta', 'lq', 'gt'])
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']
        
        fake_g_output, aligned, deform_groups = self.generator(lq, gt)
        # fake_g_output = self.generator(lq,gt)
        # flops = flops_counter.get_model_complexity_info(self.generator, (3,32,32), as_strings=False)
        
        '''
        # new dataset. Do some test at here
        lq = nib.Nifti1Image(np.squeeze(lq.cpu().detach().numpy()[:,1,:,:]), np.eye(4))
        nib.save(lq, '/homes/tzheng/code/mmediting/outputs/lq.nii.gz')
        fake_g_output = nib.Nifti1Image(np.squeeze(fake_g_output.cpu().detach().numpy()[:,0,:,:]), np.eye(4))
        nib.save(fake_g_output, '/homes/tzheng/code/mmediting/outputs/fake_g_output.nii.gz')
        gt = nib.Nifti1Image(np.squeeze(gt.cpu().detach().numpy()[:,0,:,:]), np.eye(4))
        nib.save(gt, '/homes/tzheng/code/mmediting/outputs/gt.nii.gz')
        os._exit(0)
        '''
        
        # fake_g_output = self.generator(lq)
        if fake_g_output.shape != gt.shape:
            startx = int((fake_g_output.shape[2] - gt.shape[2]) / 2)
            endx = int(startx + gt.shape[2])
            starty = int((fake_g_output.shape[3] - gt.shape[3]) / 2)
            endy = int(starty + gt.shape[3])
            fake_g_output = fake_g_output[:,:,startx:endx,starty:endy]
        
        # fake_g_output = torch.clamp(input=fake_g_output, min=-1.0, max=1.0)
        
        if torch.isnan(lq).any():
            logging.warning('nan in lq')
        elif torch.isnan(gt).any():
            logging.warning('nan in gt')
        elif torch.isnan(fake_g_output).any():
            logging.warning('nan in fake g output')


        losses = dict()
        log_vars = dict()

        # no updates to discriminator parameters.
        set_requires_grad(self.discriminator, False)
        
        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            
            # add registration loss at here
            if deform_groups is not None:
                scaled_gt = F.interpolate(gt, scale_factor=0.25, mode='trilinear')
                losses['loss_regis_pix'] = self.vxm_losses[0](scaled_gt, deform_groups[0]) * self.vxm_weights[0]
                losses['loss_regis_smooth'] = self.vxm_losses[1](torch.zeros_like(deform_groups[1]), deform_groups[1]) * self.vxm_weights[1]
                if len(deform_groups) == 3:
                    s = torch.exp(-deform_groups[2])
                    uncer_loss = self.uncerloss(torch.mul(scaled_gt, s),torch.mul(deform_groups[0], s)) + 0.02 * torch.mean(deform_groups[2])
                    losses['loss_regis_uncer'] = uncer_loss * self.vxm_weights[2]
            
            # save to have a look
            '''
            save_gt = np.squeeze(gt.cpu().detach().numpy())
            save_gt = nib.Nifti1Image(save_gt, np.eye(4))
            nib.save(save_gt, '/dataT1/Free/tzheng/mmediting_save/tmp/save_gt.nii.gz')
            deformlq = np.squeeze(deform_groups[0].cpu().detach().numpy())
            deformlq = nib.Nifti1Image(deformlq, np.eye(4))
            nib.save(deformlq, '/dataT1/Free/tzheng/mmediting_save/tmp/deformlq.nii.gz')
            deformfield = np.squeeze(deform_groups[1].cpu().detach().numpy())
            deformfield = nib.Nifti1Image(deformfield, np.eye(4))
            nib.save(deformfield, '/dataT1/Free/tzheng/mmediting_save/tmp/deformfield.nii.gz')
            uncer = np.squeeze(deform_groups[2].cpu().detach().numpy())
            uncer = nib.Nifti1Image(uncer, np.eye(4))
            nib.save(uncer, '/dataT1/Free/tzheng/mmediting_save/tmp/uncer.nii.gz')
            os._exit(0)
            '''
            
            #  uncerloss for SR
            # uncer = F.interpolate(deform_groups[2] - deform_groups[2].min() + 1, scale_factor=8, mode='trilinear')
            # losses['loss_uncer_SR'] = self.pixel_loss(torch.mul(fake_g_output, uncer), torch.mul(gt, uncer)) * 0.01
            
            # calculate flops
            # if self.udl_loss:
            #     losses['loss_udl'] = self.udl_loss(fake_g_output, gt)
            
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt)
                losses['loss_dfo'] = self.pixel_loss(aligned, F.interpolate(gt, scale_factor=0.25, mode='trilinear'))
            self.perceptual_loss = False
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style
            
            # gan loss for generator
            fake_g_pred = self.discriminator(fake_g_output)
            
            # fake_g_pred = self.discriminator(torch.unsqueeze(fake_g_output.repeat(1, 256, 1, 1), dim=1)) # torch.Size([12, 1])
            losses['loss_gan'] = self.gan_loss(
                fake_g_pred, target_is_real=True, is_disc=False)
            
            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)
            # optimize
            optimizer['generator'].zero_grad()
            if losses['loss_pix'] < 10:
                loss_g.backward()
                optimizer['generator'].step()
            else:
                print('loss_pix is too big!')
                print(losses['loss_pix'])
                losses['loss_pix'] = 0.001
                losses['loss_gan'] = 0.001
                print(gt.max())
                print(fake_g_output.max())        
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.001)
        
        # discriminator
        set_requires_grad(self.discriminator, True)
        # real
        real_d_pred = self.discriminator(gt)
        loss_d_real = self.gan_loss(
            real_d_pred, target_is_real=True, is_disc=True)
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_real=loss_d_real))
        optimizer['discriminator'].zero_grad()
        loss_d.backward()
        log_vars.update(log_vars_d)
        # fake
        fake_d_pred = self.discriminator(fake_g_output.detach())
        
        loss_d_fake = self.gan_loss(
            fake_d_pred, target_is_real=False, is_disc=True)
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_fake=loss_d_fake))
        loss_d.backward()
        log_vars.update(log_vars_d)

        if losses['loss_pix'] < 10:
            optimizer['discriminator'].step()
        else:
            losses['loss_pix'] = 0.001
            losses['loss_gan'] = 0.001
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.001)
        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))
        
        self.gt_ema = gt
        self.SR_ema = fake_g_output
        self.lq_ema = lq
        
        return outputs
