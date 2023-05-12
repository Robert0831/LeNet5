import os
from PIL import Image
import pathlib
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import cv2

class Dataset50Loader(DataLoader):
    def __init__(self,phase):
        self.data50=Dataset50(phase)
        batch_size=64
        super().__init__(dataset= self.data50,batch_size=batch_size)
        

class Dataset50(object):
    def __init__(self,dataway):
        self.path=os.path.abspath("")
        self.train=os.path.join(self.path,dataway)
        self.dataframe = pd.read_csv(self.train,sep=" ",header=None,names=["dir", "class"])
        self.len = len(self.dataframe)

    def __len__(self):
        return (self.len)
    def __getitem__(self, index):
        img_path = self.dataframe['dir'][index]
        #img= np.array(Image.open(img_path).convert("RGB"))
        #img=self.trans(image=img)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))    
        img=img/255
        img=np.transpose(img,(2,0,1))
        #labels=torch.tensor(self.dataframe['class'][index])
        labels=self.dataframe['class'][index]
        return img,labels
    
#trr=Dataset50('test.txt')
#inputs['cameraFront'][row_idx] = (self.imageFront_transform(Image.open(BytesIO(self.zz.read(wway + row['cameraFront'])))))

      

