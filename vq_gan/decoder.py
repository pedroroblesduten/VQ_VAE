import torch.nn as nn
from helper import GroupNorm, Swish, ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        def conv(in_c, out_c, kernel_size=3, stride=1, padding=1):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            return conv_layer

        def att_residual(in_c, out_c, up=True):
            att_residual_layer_up = nn.Sequential(
                ResidualBlock(in_c, out_c),
                NonLocalBlock(out_c),
                UpSampleBlock(out_c)
            )
        

            att_residual_layer = nn.Sequential(
            ResidualBlock(in_c, out_c),
            NonLocalBlock(out_c)
            )

            if up==True:
                return att_residual_layer_up
            else:
                return att_residual_layer

        def residual(in_c, out_c, up=True):
            residual_layer_dw = nn.Sequential(
                ResidualBlock(in_c, out_c),
                UpSampleBlock(out_c)
            )
            residual_layer = nn.Sequential(
            ResidualBlock(in_c, out_c)
            )

            if up==True:
                return residual_layer_dw
            else:
                return residual_layer

        self.conv1 = conv(args.latent_dim, 512)
        self.att_residual_up_128_128 = att_residual(128, 128)
        self.residual_dw_128_128 = residual(128, 128)

        self.att_residual_up_256_128 = att_residual(256, 128)
        self.residual_dw_256_128 = residual(256, 128)

        self.att_residual_up_256_256 = att_residual(256, 256)
        self.residual_dw_256_256 = residual(256, 256)

        self.att_residual_up_512_256 = att_residual(512, 256)
        self.residual_up_512_256= residual(512, 256)
        self.residual_512_256 = residual(512, 256, up=False)

        self.att_residual_up_512_512 = att_residual(512, 512)
        self.att_residual_512_512 = att_residual(512, 512, up=False)
        self.residual_up_512_512 = residual(512, 512)
        self.residual_512_512 = residual(512, 512, up=False)

        self.non_local_512 = NonLocalBlock(512)
        self.group_norm_512 = GroupNorm(512)
        self.swish = Swish()
        self.conv2 = conv(512, args.latent_dim)

    def forward(self, x):
        print('-- COMEÇANDO DECODER --')
        print(f'Shape original: {x.shape}')

        x = self.conv1(x) 
        print(f'Shape pos conv1: {x.shape}')

        x = self.residual_512_512(x)
        print(f'Shape pos 512_512: {x.shape}')

        x = self.non_local_512(x)
        print(f'Shape pos non_local_512: {x.shape}')

        x = self.residual_512_512(x)
        print(f'Shape pos 512_512: {x.shape}')
              
        x = self.att_residual_512_512(x)
        print(f'Shape pos att_residual_512_512: {x.shape}')
        
        x = self.att_residual_up_512_256(x)
        print(f'Shape pos att_residual_up_512_256: {x.shape}')
        x = self.att_residual_up_256_256(x)
        print(f'Shape pos att_residual_up_256_256: {x.shape}')

        x = self.att_residual_up_256_128(x)
        print(f'Shape pos att_residual_up_256_128: {x.shape}')
        x = self.att_residual_up_128_128(x)
        print(f'Shape pos att_residual_up_128_128: {x.shape}')


        print(f'Shape de saída do decoder')

        return x 

    
              
        

