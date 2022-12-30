import torch
import torch.nn as nn
import torchvision
from utils import load_data

class LoadDatasets():
    def __init__(self, args):
        super().__init__()
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_path = args.dataset_path   
    def a_print(self):
        print('oi')

    def returnDataset(self):
        def LoadMNIST():
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST("C:/Users/pedro/OneDrive/Área de Trabalho/classical_datasets/MNIST",
                                        train=True,
                                        download=True, 
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=self.batch_size, shuffle=True)

            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST("C:/Users/pedro/OneDrive/Área de Trabalho/classical_datasets/MNIST",
                                        train=False,
                                        download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=self.batch_size, shuffle=True)
            return train_loader

        def LoadCIFAR10():
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10("C:/Users/pedro/OneDrive/Área de Trabalho/classical_datasets/CIFAR10",
                                        train=True,
                                        download=True, 
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=self.batch_size, shuffle=True)

            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10("C:/Users/pedro/OneDrive/Área de Trabalho/classical_datasets/CIFAR10",
                                        train=False,
                                        download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=self.batch_size, shuffle=True)
            return train_loader
        
        def LoadFlowers():
            #path = r"C:\Users\pedro\OneDrive\Área de Trabalho\flowers\dandelion"
            dataset = load_data(self.data_path, self.batch_size)
            return dataset


        if self.dataset == 'MNIST':
            return LoadMNIST()
        elif self.dataset == 'CIFAR10':
            return LoadCIFAR10()
        elif self.dataset == 'flowers':
            return LoadFlowers()
