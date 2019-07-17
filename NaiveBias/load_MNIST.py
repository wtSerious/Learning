#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import struct


# In[2]:


def load_data(file_name):
    """
    To load the data of train and test
    
    Parameters
    ----------
    file_name: [4] of the  road of the data file
        [0] is the road of the train_data
        [1] is the road of the test_data
        [2] is the road of the train_labels
        [3] is the road of the test_labels
        
    Return: 
    ----------
    train_data: numpy array of shape (,)
    train_lable: list of size (,1) 
    test_data:[] numpy array of shape (,)
    test_lable:[] numpy array of size (,1)
    """
    train_data = load_image(file_name[0])
    test_data = load_image(file_name[1])
    train_labels = load_lable(file_name[2])
    test_labels = load_lable(file_name[3])
    
    return train_data,train_labels,test_data,test_labels


# MNIST 数据集的文件都是二进制格式，需要从中提取所需数据.
# 由官网可知，MNIST训练集中的
# - 第1-4个 byte 存的是文件的 magic number
# - 第5-8个 byte 存的是文件的 numbers of images
# - 第9-12个 byte 存的是图片的 number of rows
# - 第12-16个 byte 存的是图片的 number of columns
# - 从第17个 byet 开始，后面存的都是图片的像素
# 

# In[3]:


def load_image(image_file):
    """
    To load the data of Image
    
    Parameter
    --------
    image_file:str 
        the road of the image data file
        
    Return
    --------
    imgs: numpy array of shape(,)
    """
    #open the file and read the data
    f = open(image_file,'rb')
    buffers = f.read()
    
    head = struct.unpack_from('>IIII',buffers,0)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    
    bits = imgNum*width*height
    bitsString = '>' + str(bits) +'B'
    
    imgs = struct.unpack_from(bitsString,buffers,offset)
    
    f.close()
    imgs = np.reshape(imgs,[imgNum,width*height])
    
    return imgs


# In[23]:


def load_lable(file):
    """
    To load the label of Image
    
    Parameter
    ---------
    file: str 
        the road of file
        
    Return
    ---------
    lable: list of (1,)
    """
    f = open(file,'rb')
    buffers = f.read()
    
    head = struct.unpack_from('>II',buffers,0)
    
    imgNum = head[1]
    
    offset = struct.calcsize('II')
    numString = '>'+str(imgNum)+'B'
    labels = struct.unpack_from(numString,buffers,offset)
    
    f.close()
    labels = np.reshape(labels,[1,-1])
    
    return labels.flatten()
    


# In[22]:


# file = ["C:/Users/wtser/Desktop/learnData/data/Mnist/train-images.idx3-ubyte",
#        "C:/Users/wtser/Desktop/learnData/data/Mnist/t10k-images.idx3-ubyte",
#        "C:/Users/wtser/Desktop/learnData/data/Mnist/train-labels.idx1-ubyte",
#        "C:/Users/wtser/Desktop/learnData/data/Mnist/t10k-labels.idx1-ubyte"]

# train_data,train_labels,test_data,test_labels = load_data(file_name=file)

