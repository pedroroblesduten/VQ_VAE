import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import AddChannel, Compose, ScaleIntensity, ToTensor, Resize
import os
import pandas as pd
import nibabel as nib
import nilearn
import numpy as np

class LoadMRI():
    def __init__(self, args):
        super().__init__()
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



class SaveMRI():
    def __init__(self, args):
        super().__init__()

        self.data_path = args.save_path


    def saveImage(self, tensor, img_path):
        img_array = tensor[0, 0, :, :, :].to('cpu').detach().numpy()
        img_nii = nib.Nifti1Image(img_array, np.eye(4))
        save_path = os.path.join(self.data_path, img_path+'.nii.gz')
        
        nib.save(img_nii, save_path)



