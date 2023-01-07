import torch
import torch.nn as nn
import torch.nn.functional as F
from helper3D import ResidualStack3D
from monai.networks.blocks import SubpixelUpsample

class EncoderBlock3D(nn.Module):
    def __init__(self, latent_dim, i, n_l, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.n_l = n_l
        self.i = i
        self.latent_dim = latent_dim

        self._conv1 = nn.Conv3d(in_channels=1 if self.i == 0 else 144//2,
                                out_channels=144//(1 if self.i == self.n_l-1 else 2),
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                dilation=1)
        self._relu = nn.ReLU()
        self._residual_stack = ResidualStack3D(in_channels=144//(1 if self.i==self.n_l-1 else 2),
                                              res_channels=144//(1 if self.i==self.n_l-1 else 2),
                                               dropout=0.0,
                                               num_residual_layers=3)


    def forward(self, x):
        x = self._conv1(x)
        x = self._relu(x)
        x = self._residual_stack(x)
        if self.verbose:
            print(f'Shape after encoder_block: {x.shape}')
        return x

class DecoderBlock3D(nn.Module):
    def __init__(self, latent_dim, i, n_l, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.latent_dim = latent_dim
        self.i = i
        self.n_l = n_l
            
        self._residual_stack = ResidualStack3D(in_channels=144//(1 if self.i==0 else 2),
                                             res_channels=144//(1 if self.i==0 else 2),
                                             dropout=0.0,
                                             num_residual_layers=3)
        # self._upsample = SubpixelUpsample(dimensions=3,
        #                                 in_channels=self.channels[2],
        #                                  out_channels=self.channels[3],
        #                                  scale_factor=2)
        self._conv_tranpose_1 = nn.ConvTranspose3d(in_channels=144//(1 if self.i==0 else 2),
                                                   out_channels=1 if self.i==self.n_l-1 else 144//2,
                                                   kernel_size=4,
                                                   stride=2,
                                                   padding=1)
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._residual_stack(x)
        x = self._conv_tranpose_1(x)
        if self.i != self.n_l-1:
            x = self._relu(x)
        if self.verbose:
            print(f'Shape after decoder_block: {x.shape}')
        return x


class Encoder3D(nn.Module):
    def __init__(self, args, n_levels=3, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.latent_dim = args.latent_dim
        self.n_levels = n_levels
        self.encoder_blocks = nn.ModuleList()
        for i in range(n_levels): 
            self.encoder_blocks.append(EncoderBlock3D(
                self.latent_dim, i, n_levels, self.verbose))
        self.conv_out = nn.Conv3d(144, self.latent_dim,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        if self.verbose:
            print(f'Shape before encoder: {x.shape}')
        for block in self.encoder_blocks:
            x = block(x)
        x = self.conv_out(x)
        if self.verbose:
            print(f'Shape after encoder: {x.shape}')
        return x

class Decoder3D(nn.Module):
    def __init__(self, args, n_levels=3, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.latent_dim = args.latent_dim
        self.conv_in = nn.Conv3d(self.latent_dim, 144,
                                 kernel_size=3, stride=1, padding=1)
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_levels):
            self.decoder_blocks.append(DecoderBlock3D(
                self.latent_dim, i, n_levels, self.verbose))

    def forward(self, x):
        if self.verbose:
            print(f'shape de entrada do decoder: {x.shape}')
        x = self.conv_in(x)
        for block in self.decoder_blocks:
            x = block(x)
        if self.verbose:
            print(f'Shape after decoder: {x.shape}')
        return x


