#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[161]:


def loadFile(file):
    #打开一个数据文件，获取图片矩阵和label
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
        d = dict[b'data']
        #对图片矩阵进行reshape和转置，这样才能好的展示图片效果
        #cifar10中，应该先reshape(3,32（1）,32（2）),然后调换位置,(32(2),32(1),3)
        d = d.reshape(10000,3,32,32)
        d = d.transpose(0,2,3,1)
        
    return d,dict[b'labels']


# In[179]:


def load_CIFAR10():
    #获取把CIFAR10的训练集数据
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(1,6):
        dictPath = os.path.join('E:/Data/cifar-10-batches-py','data_batch_%d'%(i,))
        x,y = loadFile(dictPath)
        train_X.extend(x)
        train_Y.extend(y)
    dictPath = os.path.join('E:/Data/cifar-10-batches-py/test_batch')
    test_X ,test_Y = loadFile(dictPath)
    return train_X,train_Y,test_X,test_Y


# In[184]:


def load_Data_Lables():
    #返回的数据都是numpy的array类型
    train_x ,train_y,test_x,test_y= load_CIFAR10()
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
    return train_x,train_y,test_x,test_y


# In[183]:


# train_x,train_y,test_x,test_y = load_Data_Lables()
# classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
# numClasses = len(classes)
# sample_classNum = 7
# plt.figure(figsize=(20,20))
# for y,clas in enumerate(classes):
#     #获取在labels中与y相同的索引值
#     idxs = np.flatnonzero(labels==y)
#     #随机选择七张图片
#     idxs = np.random.choice(idxs,sample_classNum,replace=False)
    
#     for i,idx in enumerate(idxs):
#         plt_idx = i * numClasses + y + 1
#         plt.subplot(sample_classNum,numClasses,plt_idx)
#         plt.imshow(train_x[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(clas)
# plt.show()
        


# In[ ]:




