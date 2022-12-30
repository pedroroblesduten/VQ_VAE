import torch
import torch.nn as nn
import torch.nn.functional as F
from helper3d import ResidualStack
from monai.networks.blocks import SubpixelUpsample

class Encoder3D(nn.Module):
    def __init__(self, args, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.channels = [args.image_channels,
                         image_channels//2,
                         image_channels//4]
        self.image_channels = args.image_channels
        self._conv1 = nn.Conv3d(in_channels=self.channels[0],
                               out_channels=self.channels[1],
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self._conv2 = nn.Conv3d(in_channels=self.channels[1],
                               out_channels=self.channels[2],
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self._conv3 = nn.Conv3d(in_channels=self.channels[2],
                                out_channels=self.channels[2],
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self._relu = nn.ReLU()
        self._residual_stack = ResidualStack3D(in_channels=self.channels[2],
                                              res_channels=self.channels[2],
                                               dropout=0.0,
                                               num_residual_layers=3
                                               )
    def forward(self, x):
        if self.verbose:
            print(')-- STARTING ENCODER-3D --')
            print(f'Shape original: {x.shape}')
        # FIRST CONVOLUTION LAYER
        x = self._conv1(x)
        x = self._relu(x)
        if self.verbose:
            print(f'Shape after conv1: {x.shape}')
        # SECOND CONVOLUTION LAYER
        x = self._conv2(x)
        x = self._relu(x)
        if self.verbose:
            print(f'Shape after conv2: {x.shape}')
        # SEQUENCE OF RESIDUAL BLOCS
        x = self._residual_stack(x)
        if self.verbose:
            print(f'Shape after residual_stack: {x.shape}')
        # LAST CONVOLUTION LAYER
        x = self._conv3(x)
        if self.verbose:
            print(f'Shape after conv3: {x.shape}')

        return x

class Decoder3D(nn.Module):
    def __init__(self, args, verbose=False):
        super().__init__()

        self.verbose = verbose
        self.channels = [args.latent_dim, args.latent_dim*2, args.image_channels]
        self._conv1 = nn.Conv3d(in_channels=self.channels[0],
                                out_channels=self.channels[1],
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self._residual_stack = ResidualStack(in_channels=channels[1],
                                             res_channels=channels[1],
                                             dropout=0.0
                                             num_residual_layers=3)
        self._upsample = SubpixelUpsample(dimensions=3,
                                          in_channels=channels[1],
                                          out_channels=channels[2],
                                          scale_factor=)
        self._conv_tranpose_1 = nn.ConvTranspose3d(in_channels=channels[1],
                                                   out_channels=channels[2],
                                                   kernel_size=4,
                                                   stride=2,
                                                   padding=1)
    def forward(self, x):
        if self.verbose:
            print('-- COMEÇANDO DECODER 3D --')
            print(f'Shape input decoder: {x.shape}')
        # FIRST CONVOLUTION LAYER
        x = self._conv1(x)
        if self.verbose:
            print(f'Shape after conv1: {x.shape}')
        # SEQUENCE OF RESIDUAL BLOCKS
        x = self._residual_stack(x)
        if self.verbose:
            print(f'Shape after residual_stack: {x.shape}')
        # x = self._upsample(x)
        # if self.verbose(x):
        #   print(f'Shape after upsample: {x.shape}')
        x = self._conv_tranpose_1(x)
        if self.verbose(x)
            print(f'Shape after conv_tranpose: {x.shape}')

        return x



