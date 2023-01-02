import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from mrivqvae import MRI_VQVAE



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
    device = 'cuda'
    tensor = torch.randn(3, 1, 91, 109, 91).to(device)
    modelo = MRI_VQVAE(args, verbose=True)
    saida, min_indices, q_loss= modelo(tensor)
    print(f'INPUT SHAPE: {tensor.shape}')
    print(f'OUTPUT SHAPE: {saida.shape}')
    print(f'MIN INDICES: {min_indices}')
    print(f'Min_indices shape: {min_indices.shape}')
    print(f'q_loss: {q_loss}')
