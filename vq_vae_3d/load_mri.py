import monai
from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, ScaleIntensity, ToTensor
import os
import pandas as pd

class LoadMRI():
    def __init__(self, data_path):
        super().__init__()
        self.dataPath = data_path
        self.transforms = Compose([
            AddChannel(),
            ToTensor()
        ])
        

    def getImagePath(self, csv_path, separate_by_class=True):
        csv = pd.read_csv(self.csv_path)
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
            return images, None, None
            

    def loadImages(self, csv_path, separate_by_class=True):
        if separate_by_class:
            ad_imgs, cn_imgs, mci_imgs = self.getImagePath(csv_path, separate_by_class)
            ad_dataset = ImageDataset(image_files=ad_imgs, transform=self.transforms)
            cn_dataset = ImageDataset(image_files=cn_imgs, transform=self.transforms)
            mci_dataset = ImageDataset(image_files=mci_imgs, transform=self.transforms)
            return ad_dataset, cn_dataset, mci_dataset
        else:
            images, none1, none2 = self.getImagePath(csv_path, separate_by_class)
            return images, none1, none2








