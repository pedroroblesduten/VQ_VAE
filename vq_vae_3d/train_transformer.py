import os
import torch
import torch.nn as nn
import numpy as np
from load_mri import LoadMRI, SaveMRI, LoadSaveIndex
from utils import fake_dataset
from run_from_pre_trained import MriRunVQVAE
from tqdm import tqdm
from my_minGPT import GPT, GPTconfig

#Training for GPT follows: https://github.com/karpathy/nanoGPT/blob/master/train.py

class trainTransformers:
    def __init__(self, args, config)

        self.load_index = LoadSaveIndex(args)
        self.run_forward = MriRunVQVAE(args)
        self.config = config
        self.create_ckpt(args.gpt_save_ckpt)
        self.train(args)

    def create_ckpt(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def forward(self, args, run_vqvae=False):
        if run_vqvae:
            run = MriRunVQVAE(args)
        
        train_index = self.load_index(args.trans_batch_size, 'train_set')
        val_index = self.load_index(args.trans_batch_size, 'validation_set')
        data = {train_set: train_index, validation_set: val_indx}

        iter_num = 0
        best_val_loss = 1e9

        model = GPT(self.config)

        if args.gpt_load_ckpt != None :
            model.load_state_dict(args.gpt_load_ckpt)

        model.to(args.device)

        opt_dict = dict(
            learning_rate = 6e-4,
            max_iters = 600000, 
            weight_decay = 1e-2,
            betas = (0.9,0.95)
        )

        optimizer = model.configure_optimizers(opt_dict[weight_decay],
                                               opt_dict[learning_rate],
                                               opt_dict[betas])

        @torch.no_grad()
        def estimate_loss(eval_iters, data):
            out = {}

            model.eval()
            for split in ['train_set', 'validation_set']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = data[split]
                logits, loss = model(X, Y)

            out[split] = losses.mean()
            model.train()
            return out

        def get_lr(iter):
                # 1) linear warmup for warmup_iters steps
            if iter < warmup_iters:
                return learning_rate * iter / warmup_iters
            # 2) if iter > lr_decay_iters, return min learning rate
            if iter > lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
                assert 0 <= decay_ratio <= 1
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
                return min_lr + coeff * (learning_rate - min_lr)


        # --- TRAINING ---

        while True:
            if decay_lr:
                lr = get_lr(iter_num, data)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = learning_rate

            if iter_num % eval_interval == 0 and gpu_id == 0:
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                raw_model = model
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {args.gpt_save_ckpt}")
                    torch.save(checkpoint, os.path.join(args.gpt_save_ckpt))

            if iter_num == 0 and eval_only:
                break
            
            X, Y = data['train_set']

            logts, loss = model(X, Y)
            loss.backward()
            optimizer.step()

            if iter_num % log_interval == 0 and gpu_id == 0:
                lossf = loss.item() # loss as float. TODO CPU-GPU sync: profile, make sure not slow af
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

            iter_num += 1

            if iter_num > max_iters:
                break







            





                    

        








        

