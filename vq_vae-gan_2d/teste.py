from classical_datasets import LoadDatasets
import torch
import torchvision
from torchvision import utils as vutils

dataset = LoadDatasets('CIFAR10', 64)
print(dataset.dataset)
dataset.a_print()
train_loader, test_loader = dataset.returnDataset()
print(len(train_loader))
#imgs, decoded_images= next(iter(train_loader)), next(iter(test_loader))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for imgs in train_loader:
    print(imgs)
    imgs = imgs.to(device)
    break
#real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
#vutils.save_image(real_fake_images, "C:/Users/pedro/OneDrive/Área de Trabalho/git_rep_paper/vq_gan/cifar_teste.jpg"), nrow=4)
