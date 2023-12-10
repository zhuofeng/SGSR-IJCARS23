# Copyright (c) OpenMMLab. All rights reserved.
from .deepfill_disc import DeepFillv1Discriminators
from .gl_disc import GLDiscs
from .light_cnn import LightCNN
from .modified_vgg import ModifiedVGG
from .multi_layer_disc import MultiLayerDiscriminator
from .patch_disc import PatchDiscriminator
from .smpatch_disc import SoftMaskPatchDiscriminator
from .ttsr_disc import TTSRDiscriminator
from .unet_disc import UNetDiscriminatorWithSpectralNorm
from .patch_disc_custom import PatchDiscriminatorcustom
from .patch_disc_3d_custom import PatchDiscriminator3dcustom

__all__ = [
    'GLDiscs', 'ModifiedVGG', 'MultiLayerDiscriminator', 'TTSRDiscriminator',
    'DeepFillv1Discriminators', 'PatchDiscriminator', 'LightCNN',
    'UNetDiscriminatorWithSpectralNorm', 'SoftMaskPatchDiscriminator', 'PatchDiscriminatorcustom', 'PatchDiscriminator3dcustom'
]
