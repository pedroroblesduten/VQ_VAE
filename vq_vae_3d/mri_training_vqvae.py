import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from mrivqvae import MRI_VQVAE
from utils import load_data, weights_init
import torch.optim as optim
from load_mri import LoadMRI, SaveMRI

class MriTrainVQVAE:
    def __init__(self, args, verbose=False):

        self.verbose = verbose
        self.mri_vqvae = MRI_VQVAE(args, verbose=self.verbose)
        self.prepare_training_mri_vqvae()
        self.loader = LoadMRI(args)
        self.train(args)
        

    @staticmethod
    def prepare_training_mri_vqvae():
        os.makedirs('results_mri_vqvae', exist_ok=True)
        os.makedirs('checkpoints_mri_vqvae', exist_ok=True)
    
    def train(self, args, verbose=False):
        ad, cn, mci = self.loader.loadImages()
        train_dataset = cn
        print(len(cn))
        print(next(iter(cn)))
        steps_per_epoch = len(train_dataset)
        criterion = torch.nn.MSELoss()
        opt_vq = optim.Adam(
            list(self.mri_vqvae.encoder.parameters())+
            list(self.mri_vqvae.decoder.parameters())+
            list(self.mri_vqvae.codebook.parameters())+
            list(self.mri_vqvae.quant_conv.parameters())+
            list(self.mri_vqvae.post_quant_conv.parameters()),
            lr=args.learning_rate,eps=1e-8, betas=(args.beta1, args.beta2))
        print('--> STARTING VQVAE FOR BRAIN MRI <--')
        
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, min_indices, q_loss = self.mri_vqvae(imgs)
                    if self.verbose:
                        print('FIM DO VQVAE')
                        print('Indo para os calculos da loss')

                    rec_loss = criterion(imgs, decoded_images)
                    vq_loss = rec_loss + q_loss

                    opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)
                    opt_vq.step()

                    if epoch % 5 == 0 and i % 10 == 0:
                        with torch.no_grad():
                            saver = SaveMRI(args)
                            saver.saveImage(decoded_images, 'output_epoch_{epoch}')
                            VQ_LOSS = (f'E: {epoch}' + str(np.round(vq_loss.cpu().detach().numpy().item(), 5)))
                        pbar.update(0)
                torch.save(self.mri_vqvae.state_dict(), os.path.join('checkpoints_mri_vqvae', f'mri_vqvae_{epoch}.pt'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=3, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--verbose', type=str, default=False, help='Verbose to control prints in the foward pass')
    parser.add_argument('--use_ema', type=str, default=True, help='If True, use EMA for codebook update')
    parser.add_argument('--save_path', type=str, default='/save_outputs', help='Path for save autoencoder outputs')
    parser.add_argument('--csv_path', type=str, default='/scratch2/pedroroblesduten/CSV_3_CLASSES_COMPLETO_ADNI.csv')
    parser.add_argument('--dataset_path', type=str, default='/scratch2/turirezende/BRAIN_COVID/data/ADNI/images')

    args = parser.parse_args()
    args.verbose = True

    train_vqgan = MriTrainVQVAE(args, verbose=args.verbose)
    

