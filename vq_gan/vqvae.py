import torch
import torch.nn as nn
# from encoder import Encoder
# from decoder import Decoder
from codebook import Codebook
from simpler_encoder import SimplerEncoder as Encoder
from simpler_encoder import SimplerDecoder as Decoder

class VQVAE(nn.Module):
    def __init__(self, args):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

    def forward(self, imgs, verbose=False):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        if verbose:
            print(f'Shape do espaço latente antes do codebook: {quant_conv_encoded_images.shape}')
            print('Indo para codebook')
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        if verbose:
            print(f'Shape do espaço latente antes do decoder: {codebook_mapping.shape}')
            print(f'Shape dos indices: {codebook_indices.shape}')
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        if verbose:
            print('Indo para o decoder')
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss
