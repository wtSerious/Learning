#!/usr/bin/env python
# coding: utf-8
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import matplotlib.image as Mapping
from torch.utils.data import Dataset
import numpy as np
from PIL  import Image

class load_Face(Dataset):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,number=10000):
        "dir_path:文件路径"
        super(load_Face,self).__init__()
        self.data = []
        self.targets = []
        self.transform = transform
        epoch = 0
        for filename in os.listdir(root):
            if epoch >=number:
                break
            img_path = os.path.join(root,filename)
            with open(img_path,'rb') as f:
                img = Mapping.imread(f)
                self.data.append(img)
            epoch +=1
        self.targets = [1 for i in range(len(self.data))]
        self.data = np.array(self.data)
    
    def __getitem__(self,index):
        img,target = self.data[index],self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img,target
    
    def __len__(self):
        return len(self.data)


# In[78]:


