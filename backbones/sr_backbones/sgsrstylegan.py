# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import math
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from .op import fused_leaky_relu
from torch.nn import functional as F

from mmedit.models.backbones.sr_backbones.rrdb_net import RRDB
from mmedit.models.builder import build_component
from mmedit.models.common import PixelShufflePack, make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

@BACKBONES.register_module()
class SGSR_StyleGAN(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 img_channels=3,
                 rrdb_channels=64,
                 num_rrdbs=23,
                 style_channels=512,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 pretrained=None,
                 bgr2rgb=False):

        super().__init__()
        # input size must be strictly smaller than output size
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')
        
        # latent bank (StyleGANv2), with weights being fixed this is just StyleGAN
        self.generator = build_component(
            dict(
                type='StyleGANv2Generator',
                out_size=out_size,
                style_channels=style_channels,
                num_mlps=num_mlps,
                channel_multiplier=channel_multiplier,
                blur_kernel=blur_kernel,
                lr_mlp=lr_mlp,
                default_style_mode=default_style_mode,
                eval_style_mode=eval_style_mode,
                mix_prob=mix_prob,
                pretrained=pretrained,
                bgr2rgb=bgr2rgb))
        
        self.generator.requires_grad_(False)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    img_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for res in encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder.append(block)
        
        # turn encoder features into latent code use style subnet
        # style = GradualStyleBlock(512, 512, 16)
        self.styles = nn.ModuleList()
        for i in range(num_styles):
            if i < 8:
                style = GradualStyleBlock(512, 512, 4)
            elif i < 10:
                style = GradualStyleBlock(512, 512, 8)
            elif i < 12:
                style = GradualStyleBlock(512, 512, 16)
            else:
                style = GradualStyleBlock(512, 512, 32)
            self.styles.append(style)

        # additional modules for StyleGANv2
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
            if img_channels == 3:
                self.fusion_skip.append(
                    nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))
            else:
                self.fusion_skip.append(
                    nn.Conv2d(num_channels + 3, 1, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i
            for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        self.decoder = nn.ModuleList()

        for res in decoder_res:
            if res == in_size:
                in_channels = channels[res]
            else:
                in_channels = 2 * channels[res]

            if res < out_size:
                out_channels = channels[res * 2]
                self.decoder.append(
                    PixelShufflePack(
                        in_channels, out_channels, 2, upsample_kernel=3))
                print('append pixelshuffle pack')
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, 1, 3, 1, 1)))
        

    def forward(self, lq):
        """Forward function.

        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).

        Returns:
            Tensor: Output HR image.
        """
        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')

        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        
        latents = []
        for j in range(14):
            if j < 8:
                latents.append(self.styles[j](encoder_features[3])) # torch.Size([1, 512, 4, 4])
            elif j < 10:
                latents.append(self.styles[j](encoder_features[2])) # torch.Size([1, 512, 8, 8])
            elif j < 12:
                latents.append(self.styles[j](encoder_features[1])) # torch.Size([1, 512, 16, 16])
            else:
                latents.append(self.styles[j](encoder_features[0])) # torch.Size([1, 512, 32, 32])

        encoder_features = encoder_features[::-1]
        
        latent = torch.stack(latents, dim=1) # latent torch.Size([1, 14, 512])
        
        # latent = encoder_features[0].view(lq.size(0), -1, self.style_channels) # encoder_features[0] torch.Size([1, 7168])  latent torch.Size([1, 14, 512])
        encoder_features = encoder_features[1:]

        # generator
        
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]
        
        '''
        injected_noise = [
            None
            for i in range(self.generator.num_injected_noises)
        ]
        '''
        # style uses the encoder input, but noise still uses the original noise of stylegan
        # 4x4 stage
        out = self.generator.constant_input(latent) # ([1, 512, 4, 4]) this outoput is randomly sampled. refer to the original paper.
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0]) # this is a ModulatedStyleConv refer to the stylegan paper, take former input, randomly sampled and noise as input
        
        # replace out with latent feature
        # out = self.generator.conv1(encoder_features[0], latent[:, 0], noise=injected_noise[0])
        
        skip = self.generator.to_rgb1(out, latent[:, 1]) # skip:[1, 3, 4, 4]
        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2], # now the noise is randomly injected?
                self.generator.to_rgbs):

            # feature fusion by channel-wise concatenation
            if out.size(2) <= self.in_size:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]
                out = torch.cat([out, feat], dim=1) # before concact: torch.Size([1, 512, 4, 4]) after concact: torch.Size([1, 1024, 4, 4])
                out = self.fusion_out[fusion_index](out)
                skip = torch.cat([skip, feat], dim=1)
                skip = self.fusion_skip[fusion_index](skip)
            
            # original StyleGAN operations, before this has done feature fusion. Thus feature map here is different
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)
            
            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)

            _index += 2
        
        # decoder
        hr = encoder_features[-1] # hr [1, 512, 32, 32]
        for i, block in enumerate(self.decoder):
            if i > 0:    
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)
        
        return hr
    
    def forward_stylegan(self, lq):
        styles = [torch.randn((8, self.style_channels))]
        styles = [s.cuda() for s in styles]
        if len(styles) < 2:
            inject_index = 14

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        injected_noise = [
            None
            for i in range(self.generator.num_injected_noises)
        ]
        
        # 4x4 stage
        out = self.generator.constant_input(latent) # ([1, 512, 4, 4]) this outoput is randomly sampled. refer to the original paper.
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0]) # this is a ModulatedStyleConv refer to the stylegan paper, take former input, randomly sampled and noise as input
        skip = self.generator.to_rgb1(out, latent[:, 1]) # skip:[1, 3, 4, 4]
        _index = 1
        
        # 8x8 ---> higher res
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):
            # original StyleGAN operations, before this has done feature fusion. Thus feature map here is different
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)
            _index += 2

        img = skip
        return img

    def forward_onlyfromnoise(self, lq):
        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')

        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        
        encoder_features = encoder_features[::-1]
        
        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels) # encoder_features[0] torch.Size([1, 7168])  latent torch.Size([1, 14, 512])
        encoder_features = encoder_features[1:]

        # generator
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]
        # 4x4 stage
        out = self.generator.constant_input(latent) # ([1, 512, 4, 4]) this outoput is randomly sampled. refer to the original paper.
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0]) # this is a ModulatedStyleConv refer to the stylegan paper, take former input, randomly sampled and noise as input
        skip = self.generator.to_rgb1(out, latent[:, 1]) # skip:[1, 3, 4, 4]
        _index = 1
        
        # 8x8 ---> higher res
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):
            # original StyleGAN operations, before this has done feature fusion. Thus feature map here is different
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)
            _index += 2

        img = skip      
        return img


    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class RRDBFeatureExtractor(nn.Module):
    """Feature extractor composed of Residual-in-Residual Dense Blocks (RRDBs).

    It is equivalent to ESRGAN with the upsampling module removed.

    Args:
        in_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Default: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32):

        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDB,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat = self.conv_first(x)
        return feat + self.conv_body(self.body(feat))


class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )