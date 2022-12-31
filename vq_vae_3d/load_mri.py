import monai
from monai.data import ImageDaset
from monai.transforms import AddChannel, Compose, ScaleIntensity, ToTensor
import os
import pandas as pd

class LoadMRI():
    def __init__(self, data_path, csv_path):
        super().__init__()
        self.dataPath = data_path
        self.transforms = Compose([
            AddChannel(),
            ToTensor()
        ])
        self.csv_path = csv_path 

    def getImagePath(self, csv_path, separate_by_class=True):
        csv = pd.read_csv(self.csv_path)
        csv = csv.sample(frac=1)]
        if self.separate_by_class:
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
            ad_dataset = ImageDaset(image_files=self.ad_imgs, transforms=self.transforms)
            cn_dataset = ImageDaset(image_files=self.cn_imgs, transforms=self.transforms)
            mci_dataset = ImageDaset(image_files=self.mci_imgs, transforms=self.transforms)
            return ad_dataset, cn_dataset, mci_dataset
        else:
            images, none1, none2 = self.getImagePath(csv_path, separate_by_class)
            return images, none1, none2








