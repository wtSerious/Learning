#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys;
sys.path.append("E:/JupyterEnviroment/Learning/MachineLearning/Network/")
import numpy as np
from cs231n.layers import *


# In[ ]:


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
  
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C,H,W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn((H // 2)*(W // 2)*num_filters, hidden_dim)
        print('w2',self.params['W2'].shape)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        
        for k,v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    def loss(self,X,y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        scores = None
        
        conv_forward_out_1, cache_forward_1 = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        print(conv_forward_out_1.shape)
        affine_forward_out_2, cache_forward_2 = affine_forward(conv_forward_out_1, self.params['W2'], self.params['b2'])
        affine_relu_2, cache_relu_2 = relu_forward(affine_forward_out_2)
        scores, cache_forward_3 = affine_forward(affine_relu_2, self.params['W3'], self.params['b3'])
        
        if y is None:
          return scores
    
        loss, grads = 0, {}
        loss, dout = soft_loss(scores, y)

        # Add regularization
        loss += self.reg * 0.5 * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2))

        dX3, grads['W3'], grads['b3'] = affine_backward(dout, cache_forward_3)
        dX2 = relu_backward(dX3, cache_relu_2)
        dX2, grads['W2'], grads['b2'] = affine_backward(dX2, cache_forward_2)
        dX1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dX2, cache_forward_1)

        grads['W3'] = grads['W3'] + self.reg * self.params['W3']
        grads['W2'] = grads['W2'] + self.reg * self.params['W2']
        grads['W1'] = grads['W1'] + self.reg * self.params['W1']
        
        return loss, grads


# In[4]:


get_ipython().system('jupyter nbconvert --to python cnn.ipynb')

