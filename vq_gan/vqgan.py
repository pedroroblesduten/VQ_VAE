import torch
import torch.nn as nn
from simpler_encoder import SimplerEncoder as Encoder
from simpler_encoder import SimplerDecoder as Decoder
from codebook import Codebook

class VQGAN(nn.Module):
    def __init__(self, args, verbose=False):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.verbose = verbose

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

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)

        return decoded_images, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder._conv_trans_2
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        A = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        A = torch.clamp(A, 0, 1e4).detach()
        

        return 0.8*A

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

        

