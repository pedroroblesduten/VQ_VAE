import torch
import torch.nn as nn
from my_minGPT import GPT, GPTconfig
import os


out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
batch_size = 3
block_size = 320
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-2
betas = (0.9, 0.95)
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda'
# compile = True # use PyTorch 2.0 to compile the model to be faster
gpu_id = 0
if gpu_id == 0:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + gpu_id) # note: each worker gets a different seed
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


x = torch.randint(5025, (batch_size, block_size), device=device)
y = torch.randint(5025, (batch_size, block_size), device=device)
get_batch = lambda split: (x, y)

gptconf = GPTconfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layers = 1, n_heads = 2, embedding_dim = 768 # size of the mod, # for determinism
)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95))


# sequence = torch.randint(low=1, high=1025,(batch_size, block_size))

out = model.generate(x, 3)
print(out.shape)

#print(out)

