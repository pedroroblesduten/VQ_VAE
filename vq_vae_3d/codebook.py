import torch
from torch import nn
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2

class Codebook3D(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937
    Based on the original implementation by DeepMind
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    And Andrej Karpathy implementation for pytorch
    https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/quantize.py
    """
    def __init__(self, args, verbose=False):
        super(Codebook, self).__init__()
        self.verbose = verbose
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        self.verbose = verbose
        
        # Initialization of the codebook
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)
        self.register_buffer('data_initialized', torch.zeros(1))


    def forward(self, z):
        if self.verbose:
            print('-- COMEÇANDO CODEBOOK --')
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        
        # Data dependent codebook initialization follows:
        # https://arxiv.org/pdf/2005.08520.pdf
        
        if self.training and self.data_initialized.item() == 0:
            if self.verbose:
                print('Running kmeans for codebook initialization')
            rp = torch.randperm(z_flattened.size(0))
            kmeans = kmeans2(z_flattened[rp[:20000]].data.cpu().numpy(), self.num_codebook_vectors, minit='points')
            self.embedding.weight.data.copy_(torch.from_numpy(kmeans[0]))
            self.data_initialized.fill_(1)



        d = (torch.sum(input=(z_flattened**2), dim=1, keepdim=True)
            + torch.sum(input=(self.embedding.weight**2), dim=1)
            - 2*(torch.matmul(z_flattened, self.embedding.weight.t())))
    
        if self.verbose:
            print(f'Shape das distancias: {d.shape}')

        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        #e_latent_loss = F.mse_loss(z_q.detach(), z)
        e_latent_loss = (z_q.detach() - z).pow(2).mean()

        #q_latent_loss = F.mse_loss(z_q, z.detach())
        q_latent_loss = (z_q - z.detach()).pow(2).mean()

        loss = q_latent_loss + self.beta*e_latent_loss
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)
        if self.verbose:
            print(f'Shape do espaço latente de saida do codebook: {z_q.shape}')
            print('-- FIM DO CODEBOOK --')

        return z_q, min_encoding_indices, loss

# EMA for codebook update instead of an auxiliary loss.
# Some works shows that EMA convergers faster and are independet of the choice of the optimizer
# Implementation follows: https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fWmjFfa8U5NI

class CodebookEMA3D(nn.Module):
    def __init__(self, args, verbose=False):
        super(CodebookEMA, self).__init__()

        self.verbose = verbose
        #Initialization of the codeebooks
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        
        # Initialization of the codebook
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.normal_()
        
        self.register_buffer('_ema_cluster_size', torch.zeros(self.num_codebook_vectors))
        self._ema_w = nn.Parameter(torch.Tensor(self.num_codebook_vectors, self.latent_dim))
        self._ema_w.data.normal_()

        self._decay = 0.99
        self._epsilson = 1e-5

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_shape = z.shape
        z_flattened = z.view(-1, self.latent_dim)


        d = (torch.sum(input=(z_flattened**2), dim=1, keepdim=True)
            + torch.sum(input=(self.embedding.weight**2), dim=1)
            - 2*(torch.matmul(z_flattened, self.embedding.weight.t())))

        encodings_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = torch.zeros(encodings_indices.shape[0], self.num_codebook_vectors, device=z.device)
        encodings.scatter_(1, encodings_indices, 1)



        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        
        if self.training:
            if self.verbose:
                print('Starting EMA update')
            self._ema_cluster_size = ((self._ema_cluster_size*self._decay)
                                    + (1-self._decay)*(torch.sum(encodings, 0)))
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilson)
                / (n + self.num_codebook_vectors*self._epsilson)*n
            )
            dw = torch.matmul(encodings.t(), z_flattened)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self.embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))


        e_latent_loss = F.mse_loss(z_q.detach(), z)
        # q_latent_loss = F.mse_loss(z_q, z.detach())
        loss = self.beta*e_latent_loss
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)
        if self.verbose:
            print(f')Shape do espaço latente de saida do codebook: {z_q.shape}')
            print('-- FIM DO CODEBOOK --')

        return z_q, encodings_indices, loss
