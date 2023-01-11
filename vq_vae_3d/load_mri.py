import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import AddChannel, Compose, ScaleIntensity, ToTensor, Resize
import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch

class LoadMRI:
    def __init__(self, args):
        self.args = args
        self.dataPath = args.dataset_path
        self.batch_size = args.batch_size
        self.transforms = Compose([
            AddChannel(),
            ToTensor()
        ])
        

    def getImagePath(self, separate_by_class=True):
        csv = pd.read_csv(self.args.csv_path)
        csv = csv.sample(frac=1)
        if separate_by_class:
            ad = csv.loc[csv['Group'] == 'AD']
            ad_imgs =[os.sep.join([self.dataPath, f]) for f in list(ad['folder'])]
            


            cn = csv.loc[csv['Group'] == 'CN']
            cn_imgs = [os.sep.join([self.dataPath, f]) for f in list(cn['folder'])]

            mci = csv.loc[csv['Group'] == 'MCI']
            mci_imgs = [os.sep.join([self.dataPath, f]) for f in list(mci['folder'])]

            return ad_imgs, cn_imgs, mci_imgs

        else:
            images = [os.sep.join([self.dataPath, f]) for f in list(csv['folder'])]
            return images
            

    def loadImages(self, separate_by_class=True):
        if separate_by_class:
            ad_imgs, cn_imgs, mci_imgs = self.getImagePath()

            ad_dataset = ImageDataset(image_files=ad_imgs, transform=self.transforms)
            ad_dataloader = DataLoader(ad_dataset, batch_size=self.batch_size)

            cn_dataset = ImageDataset(image_files=cn_imgs, transform=self.transforms)
            cn_dataloader = DataLoader(cn_dataset, batch_size=self.batch_size)

            mci_dataset = ImageDataset(image_files=mci_imgs, transform=self.transforms)
            mci_dataloader = DataLoader(mci_dataset, batch_size=self.batch_size)
            return ad_dataloader, cn_dataloader, mci_dataloader

        else:
            images = self.getImagePath(separate_by_class)
            imgs_dataset = ImageDataset(image_files=images, transform=self.transforms)
            imgs_dataloader = DataLoader(imgs_dataset, batch_size=self.batch_size)
            return imgs_dataloader



class SaveMRI:
    def __init__(self, args):

        self.data_path = args.save_path
        self.save_img_mode = args.save_mode


    def saveImage(self, tensor, img_path):
        img_array = tensor[0, 0, :, :, :].to('cpu').detach().numpy()
        img_nii = nib.Nifti1Image(img_array, np.eye(4))

        if self.save_img_mode == 'run' or 'Run':
            repo_name = "forward_results"

        elif self.save_img_mode == 'training' or 'Training':
            repo_name = 'training_results'

        elif self.save_img_mode == 'corrected' or 'Corrected':
            repo_name = 'corrected_results'

        full_path = os.path.join(self.save_path, repo_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        save_path = os.path.join(full_path, img_path + '.nii.gz')
            
        nib.save(img_nii, save_path)


class LoadSaveIndex():
    def __init__(self, args, batch_size=1):
        
        self.index_path = args.index_path
        self.batch_size = batch_size
        self.create_path(self.index_path)

    def create_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def saveIndex(self, list_of_tensors, index_set):
        
        if index_set == 'train_set':
            path = os.path.join(self.index_path, 'train_set')
        elif index_set == 'validation_set':
            path = os.path.join(self.index_path, 'validation_set')
        

        if not os.path.exists(path):
            os.makedirs(path)
        
        for b, batch in enumerate(list_of_tensors):
            array = batch.cpu().numpy()

            path_save = os.path.join(path, f"idx_array_img_{b}.npy")
            np.save(path_save, array.flatten())

    def loadIndex(self, batch_size, index_set):
        if index_set == 'train_set':
            path = os.path.join(self.index_path, 'train_set')
        elif index_set == 'validation_set':
            path = os.path.join(self.index_path, 'validation_set')
        
        arrays_files = os.listdir(path)
        list_of_arrays = []
        for file in arrays_files:
            array = np.load(os.path.join(path, file))
            
            list_of_arrays.append(torch.tensor(array))

        tensor_of_tensors = torch.stack(list_of_arrays)
        #dataset = torch.utils.data.TensorDataset(*map(torch.from_numpy, list_of_arrays))

        dataloader = DataLoader(tensor_of_tensors, batch_size=batch_size, shuffle=True)
        return dataloader

            
            


        



        
        

    




