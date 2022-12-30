from torch import nn
from helper import GroupNorm, Swish, ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 256

        def conv(in_c, out_c, kernel_size=3, stride=1, padding=1):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_c,
                           out_c,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
            )
            return conv_layer

        def att_residual(in_c, out_c, dw=True):
            att_residual_layer_dw = nn.Sequential(
                ResidualBlock(in_c, out_c),
                NonLocalBlock(out_c),
                DownSampleBlock(out_c)
            )
        

            att_residual_layer = nn.Sequential(
            ResidualBlock(in_c, out_c),
            NonLocalBlock(out_c)
            )

            if dw==True:
                return att_residual_layer_dw
            else:
                return att_residual_layer

        def residual(in_c, out_c, dw=True):
            residual_layer_dw = nn.Sequential(
                ResidualBlock(in_c, out_c),
                DownSampleBlock(out_c)
            )
            residual_layer = nn.Sequential(
            ResidualBlock(in_c, out_c)
            )

            if dw==True:
                return residual_layer_dw
            else:
                return residual_layer

        self.conv1 = conv(args.image_channels, 128)
        self.att_residual_dw_128_128 = att_residual(128, 128)
        self.residual_dw_128_128 = residual(128, 128)

        self.att_residual_dw_128_256 = att_residual(128, 256)
        self.residual_dw_128_256 = residual(128, 256)

        self.att_residual_dw_256_256 = att_residual(256, 256)
        self.residual_dw_256_256 = residual(256, 256)

        self.att_residual_dw_256_512 = att_residual(256, 512)
        self.residual_dw_256_512 = residual(256, 512)
        self.residual_256_512 = residual(256, 512, dw=False)

        self.att_residual_dw_512_512 = att_residual(512, 512)
        self.residual_dw_512_512 = residual(512, 512)
        self.residual_512_512 = residual(512, 512, dw=False)

        self.non_local_512 = NonLocalBlock(512)
        self.group_norm_512 = GroupNorm(512)
        self.swish = Swish()
        self.conv2 = conv(512, args.latent_dim)

    def forward(self, x):
        print('-- COMEÇANDO ENCODER --')
        print(f'Shape original: {x.shape}')

        x = self.conv1(x) 
        print(f'Shape pos conv1: {x.shape}')

        # x = self.residual_dw_128_128(x)
        # print(f'Shape pos 128_128: {x.shape}')

        x = self.residual_dw_128_128(x)
        print(f'Shape pos 128_128: {x.shape}')

        x = self.residual_dw_128_256(x)
        print(f'Shape pos 128_256: {x.shape}')

        # x = self.residual_dw_256_256(x)
        # print(f'Shape pos 256_256: {x.shape}')

        x = self.residual_256_512(x)
        print(f'Shape pos 256_512: {x.shape}')

        x = self.residual_512_512(x)
        print(f'Shape pos 512_512: {x.shape}')

        x = self.non_local_512(x)
        print(f'Shape pos non_local_512: {x.shape}')

        x = self.residual_512_512(x)
        print(f'Shape pos 512_512: {x.shape}')

        x = self.group_norm_512(x)
        print(f'Shape pos group_norm_512: {x.shape}')

        x = self.swish(x)
        print(f'Shape pos Swish: {x.shape}')

        x = self.conv2(x)
        print(f'Shape pos conv2: {x.shape}')

        print(f'Shape de saída do encoder: {x.shape}')
        print('-- FIM DO ENCODER --')

        return x

        





