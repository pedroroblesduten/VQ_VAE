import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqvae import VQVAE
from utils import load_data, weights_init
import torch.optim as optim
from classical_datasets import LoadDatasets

class TrainVQVAE:
    def __init__(self, args, verbose=False):
        self.verbose = verbose
        self.vqvae = VQVAE(args, verbose=self.verbose).to(device=args.device)
        self.prepare_training_vqvae()
        self.train(args)
        self.opt_vq = optim.Adam(
            list(self.vqvae.encoder.parameters())+
            list(self.vqvae.decoder.parameters())+
            list(self.vqvae.codebook.parameters())+
            list(self.vqvae.quant_conv.parameters())+
            list(self.vqvae.post_quant_conv.parameters()),
            lr=args.learning_rate,eps=1e-8, betas=(args.beta1, args.beta2)
        )

    @staticmethod
    def prepare_training_vqvae():
        os.makedirs('results_vqvae', exist_ok=True)
        os.makedirs('checkpoints_vqvae', exist_ok=True)

    def train(self, args, verbose=False):
        criterion = torch.nn.MSELoss()
        opt_vq = optim.Adam(
            list(self.vqvae.encoder.parameters())+
            list(self.vqvae.decoder.parameters())+
            list(self.vqvae.codebook.parameters())+
            list(self.vqvae.quant_conv.parameters())+
            list(self.vqvae.post_quant_conv.parameters()),
            lr=args.learning_rate,eps=1e-8, betas=(args.beta1, args.beta2))

        train_dataset, _= LoadDatasets(args.dataset, args.batch_size).returnDataset()
        if args.dataset == 'MNIST' or args.dataset == 'CIFAR10':
            train_dataset = iter(train_dataset)
        steps_per_epoch = len(train_dataset)

        
        print('Começando o treino do VQVAE')
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (imgs, label) in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqvae(imgs)
                    if self.verbose:
                        print('FIM DO VQVAE')
                    

                    rec_loss = criterion(imgs, decoded_images)
                    vq_loss = rec_loss + q_loss

                    opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)
                    opt_vq.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join('results_vqvae', f'{epoch}_{i}.jpg'), nrow=4)

                    pbar.set_postfix(
                        VQ_loss=np.round(vq_loss.cpu().detach().numpy().item(), 5))
                    pbar.update(0)
                    

                torch.save(self.vqvae.state_dict(), os.path.join('checkpoints_vqvae', f'vqgan_epoch_{epoch}.pt'))
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch_size', type=int, default=10, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--verbose', type=str, default=False, help='Verbose to control prints in the foward pass')
    parser.add_argument('--use_ema', type=str, default=False, help='If True, use EMA for codebook update')
    parser.add_argument('--dataset', type=str, default='flowers', help='If True, use EMA for codebook update')

    args = parser.parse_args()
    args.dataset_path = r"C:\Users\pedro\OneDrive\Área de Trabalho\flowers\rose"
    # args.verbose = True
    args.use_ema = True
    args.dataset = 'CIFAR10'

    train_vqgan = TrainVQVAE(args, verbose=args.verbose)

