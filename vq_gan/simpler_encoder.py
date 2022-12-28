from torch import nn
from helper import GroupNorm, Swish, ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock
from simpler_helper import ResidualStack
import torch.nn.functional as F

class SimplerEncoder(nn.Module):
    def __init__(self, args):
        super(SimplerEncoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=args.image_channels,
                                 out_channels=64,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=64,
                                 out_channels=128,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=128,
                                 out_channels=256,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=256,
                                             num_hiddens=256,
                                             num_residual_layers=3,
                                             num_residual_hiddens=256)

    def forward(self, x, verbose=False):
        if verbose:
            print('-- COMEÇANDO ENCODER --')
            print(f'Shape original: {x.shape}')

        x = self._conv_1(x)
        if verbose:
            print(f'Shape pos conv1: {x.shape}')
        x = F.relu(x)
        
        x = self._conv_2(x)
        if verbose:
            print(f'Shape pos _conv_2: {x.shape}')
        x = F.relu(x)
        
        x = self._conv_3(x)
        if verbose:
            print(f'Shape pos _conv_3: {x.shape}')

        x = self._residual_stack(x)
        if verbose:
            print('Shape de saida do encoder: {x.shape}')
            print('-- FIM DO ENCODER --')
        return x

class SimplerDecoder(nn.Module):
    def __init__(self, args):
        super(SimplerDecoder, self).__init__()
            
        self._conv_1 = nn.Conv2d(in_channels=args.latent_dim,
                                     out_channels=256,
                                     kernel_size=3, 
                                     stride=1, padding=1)
            
        self._residual_stack = ResidualStack(in_channels=256,
                                                 num_hiddens=256,
                                                 num_residual_layers=3,
                                                 num_residual_hiddens=256)
            
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=256,
                                                    out_channels=128,
                                                    kernel_size=4, 
                                                    stride=2, padding=1)
            
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=128, 
                                                    out_channels=3,
                                                    kernel_size=4, 
                                                    stride=2, padding=1)

    def forward(self, x, verbose=False):
        if verbose:
            print('-- COMEÇANDO DECODER --')
            print(f'Shape de entrada do decoder: {x.shape}')
        x = self._conv_1(x)
        if verbose:
            print(f'Shape pos conv1: {x.shape}')
            
        x = self._residual_stack(x)
        if verbose:
            print(f'Shape pos Residual: {x.shape}')
            
        x = self._conv_trans_1(x)
        if verbose:
            print(f'Shape pos _conv_trans_1: {x.shape}')
        x = F.relu(x)

        x = self._conv_trans_2(x)
        if verbose:
            print(f'Shape pos _conv_trans_2: {x.shape}')

            print('-- FIM DO DECODER --')
            
        
        return x
