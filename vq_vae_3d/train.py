import torch
import torch.nn as nn
import os
import numpy
from my_minGPT import GPT, GPTconfig
from tqdm import tqdm

class TrainGPT():
    def __init__(self, config)
    
        self.config = config
        self.model = GPT(config)
        self.prepare_training_gpt()
        self.loader = LoadVQResults()
        self.saver = SaveResults()
        self.train()

    @staticmethod
    def prepare_training_gpt():
        os.makedirs('results_transformer', exist_ok=True)
        os.makedirs('checkpoints_transformer', exist_ok=True)


    def train(self, config):
        index = self.loader.loadIndex()
        steps_per_epoch = len(index)
        optimizer = self.model.configure_optimizers(weight_decay=1e-2,
                                                    learning_rate=1e-4,
                                                    betas=(0.9, 0.95))
        print('--> STARTING TRANSFORMER TRAINING <--')
        
        for epoch in range(config.epochs):
            with tqdm(range(len(index))) as pbar:
                for i, idx in zip(pbar, train_dataset):
                    idx = idx.to(device=config.device)
                    logits, loss = self.model(idx)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if epoch%5==0 and i%20==0:
                        with torch.no_grad():
                            self.saver.saveIndex(os.path.join(config.save_path, f'transformer_E_{epoch}_B_{i}'))
                    LOSS = (f'E: {epoch}' + str(np.round(vq_loss.cpu().detach().numpy().item(), 5)))
                    pbar.update(0)
                torch.save(self.mri_vqvae.state_dict(), os.path.join('checkpoints_transformer', f'transformer_{epoch}.pt'))

if __name__ == '__main__':
    gptconf = GPTConfig(block_size = block_size,
                        n_layers = 12,
                        n_heads = 12,
                        embedding_dim = 768)


    train_gpt = TrainGPT(gptconf)
                    

