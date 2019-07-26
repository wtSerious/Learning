#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pickle
import numpy as np
import os
import sys
from scipy.misc import imread
from torch.utils.data import Dataset
from PIL import Image


# In[22]:


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    
class local_CIFAR10(Dataset):
    base_folder = 'cifar-10-batches-py'
    train_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5']
    
    test_list = ['test_batch']
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888'}   
    def __init__(self, root, train=True,transform=None, target_transform=None,download=False):
        super(local_CIFAR10, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.root = root
        if self.train:
            self.download_list = self.train_list
        else:
            self.download_list = self.test_list
            
        self.data = []
        self.targets = []
        
        for file_name in self.download_list:
            file_path = os.path.join(root,self.base_folder,file_name)
            with open(file_path,'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f,encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                    
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        
        self._load_meta()
        
    def _load_meta(self):
        path = os.path.join(self.root,self.base_folder,self.meta['filename'])
        with open(path,'rb') as infile:
            data = pickle.load(infile)
            self.classes = data[self.meta['key']]
        self.classes_to_idx = {_class: i for i ,_class in enumerate(self.classes)}
    
    def __getitem__(self,index):
        img, target = self.data[index],self.targets[index]
        
        img = Image.fromarry(img)
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img,target
    
    def __len__(self):
        return len(self.data)


# In[26]:


get_ipython().system('jupyter nbconvert --to python local_CIFAR10.ipynb')

