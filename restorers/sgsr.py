# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
import pdb
import numpy as np
from PIL import Image
import random
import nibabel as nib
import os
from os import listdir
import torch
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import gaussian_filter
import mmcv

from mmedit.core import tensor2img
from ..registry import MODELS
from .srgan import SRGAN


@MODELS.register_module()
class SGSR(SRGAN):

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained=pretrained)

    def testOASIScase(self):
        # just output a case at here (comment after it)
        datapath = './MRI/MRI1.nii.gz'
        dataarr = np.squeeze(nib.load(datapath).get_fdata())

        lrs = []
        hrs = []
        srs = []

        with torch.no_grad():
            for i in range(int(dataarr.shape[2]*0.2), int(dataarr.shape[2]*0.8)):
                slice = dataarr[:,:,i-1:i+2].transpose((2,0,1))
                # slice = dataarr[:,:,i]
                slice = slice[np.newaxis,:]
                # slice = np.repeat(slice, 3, axis=1)
                lr_img1 = zoom(slice, (1,1,0.125,0.125), order=2,mode='reflect')
                # lr_img1 = np.repeat(lr_img1, 3, axis=1)
                lr_img1 = torch.from_numpy(lr_img1.astype(np.float32)).clone().cuda()
                SR = self.generator(lr_img1)
                lrs.append(lr_img1[0,0,:].detach().cpu().numpy())
                hrs.append(slice[0,0,:])
                srs.append(SR[0,0,:].detach().cpu().numpy())

        lrs = np.squeeze(np.array(lrs)).astype('float32')
        hrs = np.squeeze(np.array(hrs)).astype('float32')
        srs = np.squeeze(np.array(srs)).astype('float32')
        lrs = nib.Nifti1Image(lrs, np.eye(4))
        nib.save(lrs, '/path/lr.nii.gz')
        hrs = nib.Nifti1Image(hrs, np.eye(4))
        nib.save(hrs, '/path/tmp/hr.nii.gz')
        srs = nib.Nifti1Image(srs, np.eye(4))
        nib.save(srs, '/path/sr.nii.gz')
        os._exit(0)

    def testmanyOASIScases(self):
        datapath = '/path/hr_'
        fixedfilenames = []
        for i in range(0,100):
            fixedfilenames.append(datapath + str(i) + '.nii.gz')
        
        casenum = 0
        for filename in fixedfilenames:
            dataarr = np.squeeze(nib.load(filename).get_data())
            lrs = []
            hrs = []
            srs = []
            # lqs = []
            with torch.no_grad():
                for i in range(0, int(dataarr.shape[0])):
                    # slice = dataarr[:,:,i-1:i+2].transpose((2,0,1))
                    slice = dataarr[i,:,:]
                    slice = slice[np.newaxis,np.newaxis,:]
                    slice = np.repeat(slice, 3, axis=1)
                    lr_img1 = zoom(slice, (1,1,0.125,0.125), order=2,mode='reflect')
                    lr_img1 = torch.from_numpy(lr_img1.astype(np.float32)).clone().cuda()
                    slice = torch.from_numpy(slice.astype(np.float32)).clone().cuda()
                    # SR, lq = self.generator(lr_img1, slice)
                    SR = self.generator(lr_img1)
                    lrs.append(lr_img1[0,1,:].detach().cpu().numpy())
                    hrs.append(slice[0,1,:].detach().cpu().numpy())
                    srs.append(SR[0,0,:].detach().cpu().numpy())
                    # lqs.append(deformed[0,1,:].detach().cpu().numpy())
                    # lqs.append(lq[0,1,:].detach().cpu().numpy())

            lrs = np.squeeze(np.array(lrs)).astype('float32')
            hrs = np.squeeze(np.array(hrs)).astype('float32')
            srs = np.squeeze(np.array(srs)).astype('float32')
            # lqs = np.squeeze(np.array(lqs)).astype('float32')
            lrs = nib.Nifti1Image(lrs, np.eye(4))
            nib.save(lrs, '/path/lr_{}.nii.gz'.format(str(casenum)))
            hrs = nib.Nifti1Image(hrs, np.eye(4))
            nib.save(hrs, '/path/tmp/hr_{}.nii.gz'.format(str(casenum)))
            srs = nib.Nifti1Image(srs, np.eye(4))
            nib.save(srs, '/path/sr_{}.nii.gz'.format(str(casenum)))
            # lqs = nib.Nifti1Image(lqs, np.eye(4))
            # nib.save(lqs, '/dataT1/Free/tzheng/mmediting_save/SROASIS/oasis_deform/lq_{}.nii.gz'.format(str(casenum)))
            print('Finished {}'.format(casenum))
            casenum += 1
        os._exit(0)
            
    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        # self.testmanyOASIScases()
        self.testOASIScase()
        
        output, _, _ = self.generator(lq,gt)
        # normalize from [-1, 1] to [0, 1]
        output = (output + 1) / 2.0

        if gt is not None and output.shape != gt.shape:
            startx = int((output.shape[2] - gt.shape[2]) / 2)
            endx = int(startx + gt.shape[2])
            starty = int((output.shape[3] - gt.shape[3]) / 2)
            endy = int(starty + gt.shape[3])
            output = output[:,:,startx:endx,starty:endy]

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            gt = (gt + 1) / 2.0  # normalize from [-1, 1] to [0, 1]
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()
        
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results, gt.cpu(), output.cpu(), lq.cpu()

    def forward_example(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        self.testOASIScase()
        os._exit(0)