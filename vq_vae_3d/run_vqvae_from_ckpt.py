import torch
import torch.nn as nn
import argparse
import numpy as np
from from load_mri import LoadMRI, SaveMRI
import os

class MriRunVQVAE:
    def __init__(self, args, verbose=True):
        
        self.verbose = verbose
        self.prepare_saving_index()
        self.mri_vqvae = MRI_VQVAE(args, verbose=self.verbose)
        
        self.loader_mri = LoadMRI(args)
        self.saver_mri = 
        self.forward_run(args)

    @staticmethod
    def prepare_saving_index():
        os.makedirs('index_mri_vqvae', exist_ok=True)
        os.makedirs('images_generate', exist_ok=True)


    def forward_run(self, args, verbose=False):
        self.mri_vqvae.load_state_dict(torch.load(args.load_checkpoint))
        self.mri_vqvae.eval()
        
        mri_imgs = self.loader_mri.loadImages(separate_by_class=True)
        steps_per_epoch = len(mri_imgs)
        with tqdm(range(steps_per_epoch)) as pbar:
            for i, imgs in zip(pbar, mri_imgs):
                imgs = imgs.to(args.device)[:, :, :88, :104, :88]
                decoded_imgs, index, _, self.mri_vqvae(imgs)
                self.saver_index(index)
                self.saveImage(decoded_images, f'output_img_{i}')
            pbar.update(0)

if __name__ == '__main__'
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 6)')
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
    parser.add_argument('--save_path', type=str, default='./results_mri_vqvae', help='Path for save autoencoder outputs')
    parser.add_argument('--csv_path', type=str, default='/scratch2/pedroroblesduten/CSV_3_CLASSES_COMPLETO_ADNI.csv')
    parser.add_argument('--dataset_path', type=str, default='/scratch2/turirezende/BRAIN_COVID/data/ADNI/images')

    args = parser.parse_args()
    # args.verbose = True

    run_vqgan = MriRunVQVAE(args, verbose=args.verbose)

        

