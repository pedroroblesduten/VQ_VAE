import torch
import torch.nn as nn
import argparse
import numpy as np
from load_mri import LoadMRI, SaveMRI, LoadSaveIndex
import os
from utils import fake_dataset
from mrivqvae import MRI_VQVAE
from tqdm import tqdm
from my_minGPT import GPT, GPTconfig
class MriRunVQVAE:
    def __init__(self, args, verbose=True):
        
        self.verbose = verbose
        self.mri_vqvae = MRI_VQVAE(args, verbose=self.verbose)
        
        self.loader_mri = LoadMRI(args)
        self.saver_mri = SaveMRI(args)
        self.saver_index = LoadSaveIndex(args)
        self.run_batch_size = 1
        self.forward_run(args)

    def forward_run(self, args, verbose=False):
        if args.run_from_pre_trained:
            self.mri_vqvae.load_state_dict(torch.load(args.ckpt_path))

        self.mri_vqvae.eval()
        all_index = []
        #mri_imgs = self.loader_mri.loadImages(separate_by_class=True)
        mri_imgs = fake_dataset(args.batch_size)
        steps_per_epoch = len(mri_imgs)

        with tqdm(range(steps_per_epoch)) as pbar:
            for i, imgs in zip(pbar, mri_imgs):

                imgs = imgs.to(args.device)[:, :, :88, :104, :88]
                decoded_imgs, index, _, = self.mri_vqvae(imgs)
                all_index.append(index)
                #self.saveImage(decoded_images, f'output_img_{i}')

            pbar.update(0)

        self.saver_index.saveIndex(all_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=2, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--verbose', type=bool, default=False, help='Verbose to control prints in the foward pass')
    parser.add_argument('--use_ema', type=bool, default=True, help='If True, use EMA for codebook update')

    parser.add_argument('--save_path', type=str, default='./results_mri_vqvae', help='Path for save autoencoder outputs')
    parser.add_argument('--csv_path', type=str, default='/scratch2/pedroroblesduten/CSV_3_CLASSES_COMPLETO_ADNI.csv')
    parser.add_argument('--dataset_path', type=str, default='/scratch2/turirezende/BRAIN_COVID/data/ADNI/images')
    parser.add_argument('--ckpt_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/VQVAE/') #TODO conferir checkpoint
    parser.add_argument('--train_from_pre_trained', type=bool, default=False)
    parser.add_argument('--run_from_pre_trained', type=bool, default=False)
    parser.add_argument('--index_path', type=str, default="C:/Users/pedro/OneDrive/Área de Trabalho/save_index")
    parser.add_argument('--save_mode', type=str, default='run', help='run or training or corrected')


    args = parser.parse_args()
    args.verbose = True

    run_vqgan = MriRunVQVAE(args, verbose=args.verbose)
    loader = LoadSaveIndex(args)
    index = loader.loadIndex(2)
    for x in index:
        print(x.shape)

    gptconf = GPTconfig(
    block_size = 1573, # how far back does the model look? i.e. context size
    n_layers = 1, n_heads = 2, embedding_dim = 768 # size of the mod, # for determinism
)
    device = 'cuda'
    model = GPT(gptconf)
    model.to(device)

    optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4,betas=(0.9, 0.95))


# sequence = torch.randint(low=1, high=1025,(batch_size, block_size))

    out = model.generate(index, 3)
    print(out.shape)

    print(out)


        

