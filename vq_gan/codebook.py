import torch
from torch import nn
import torch.nn.functional as F

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        
        # Initialization of the codebook
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)


    def forward(self, z, verbose=False):
        if verbose:
            print('-- COMEÇANDO CODEBOOK --')
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = (torch.sum(input=(z_flattened**2), dim=1, keepdim=True)
            + torch.sum(input=(self.embedding.weight**2), dim=1)
            - 2*(torch.matmul(z_flattened, self.embedding.weight.t())))
    
        if verbose:
            print(f'Shape das distancias: {d.shape}')

        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        loss = e_latent_loss + self.beta*q_latent_loss
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)
        if verbose:
            print(f'Shape do espaço latente de saida do codebook: {z_q.shape}')
            print('-- FIM DO CODEBOOK --')

        return z_q, min_encoding_indices, loss
