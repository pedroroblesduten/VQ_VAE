import torch
import torch.nn as nn
# from encoder import Encoder
# from decoder import Decoder
from codebook import Codebook3D
from encoder_decoder_3D import Encoder3D, Decoder3D 


class MRI_VQVAE(nn.Module):
    def __init__(self, args, verbose=False):
        super(VQVAE, self).__init__()
        self.verbose = verbose
        self.use_ema = args.use_ema

        self.encoder = Encoder3D(args, verbose=self.verbose).to(device=args.device)
        self.decoder = Decoder3D(args, verbose=self.verbose).to(device=args.device)
        if self.use_ema:
            self.codebook = CodebookEMA3D(args, verbose=self.verbose).to(device=args.device)
        else:
            self.codebook = Codebook3D(args, verbose=self.verbose).to(device=args.device)
        self.quant_conv = nn.Conv3d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv3d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        if self.verbose:
            print(f'Shape do espaço latente antes do codebook: {quant_conv_encoded_images.shape}')
            print('Indo para codebook')
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        if self.verbose:
            print(f'Shape do espaço latente antes do decoder: {codebook_mapping.shape}')
            print(f'Shape dos indices: {codebook_indices.shape}')
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        if self.verbose:
            print('Indo para o decoder')
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss
