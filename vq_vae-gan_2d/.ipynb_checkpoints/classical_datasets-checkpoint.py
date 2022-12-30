import torch
import torch.nn as nn
import torchvision

class LoadDatasets():
    def __init__(self, dataset, batch_size=16):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        # self.batch_size = batch_size
   
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
            return train_loader, test_loader

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
            return train_loader, test_loader

        if self.dataset == 'MNIST':
            return LoadMNIST()
        if self.dataset == 'CIFAR10':
            return LoadCIFAR10()
