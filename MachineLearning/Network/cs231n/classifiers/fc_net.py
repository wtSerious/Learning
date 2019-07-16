#!/usr/bin/env python
# coding: utf-8

# In[18]:


import sys;
sys.path.append("E:/JupyterEnviroment/Learning/MachineLearning/Network/")
import numpy as np
from cs231n.layers import *


# In[29]:


class TwoLayerNet(object):
    """
    A two-layer fully-connetcted neural network 
    with Relu nonlinearity and soft_loss that uses
    a modular layer design. we assum an input dimension
    of D , a hidden dimension of H, and perform classification
    over C classes.
    
    The architecure should be affine-relu-affine-softmax.
    
    """
    def __init__(self,input_dim = 3*32*32,hidden_dim = 100,
                num_classes = 10,weight_scale=1e-3,reg=0.0):
        """
         Initialize the netwrok.
     
         Parameters:
         -----------
         input_dim: An interger giving the size of the input
         hidden_dim: An interger giving the size of the hidden  layers
         num_classes: An integer giving the number of classes to classify
         weight_scale: Scalar giving the standard deviation for random
         reg: Scalar gibing L2 regularization strenth.
         """   
        self.params = {}
        self.reg = reg
        
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
    
    def loss(self,X,y=None):
       """
       Compute loss and gardient for a minibacht of data.
       
       Parameters:
       -----------
       X: A numpy array of input data, of shape(,)
       y: A Array of labels
       
       Returns:
       -----------
       if y is not None, then rum a training-time forward
       and backward pass and return a tuple of (loss,grads)
       and self.params
       """ 
       ar1_out,ar1_cache = affine_relu_forward(X,self.params['W1'],self.params['b1'])
       a2_out,a2_cache = affine_forward(ar1_out,self.params['W2'],self.params['b2'])
       scores = a2_out
        
        #if y is None then we are in test mode so just return scores
       if y is None:
        
        return scores
        
       loss,grads = 0,{}
       loss,dscores = soft_loss(scores,y)
       loss = loss + 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])
       dx2,dw2,db2 = affine_backward(dscores,a2_cache)
       grads['W2'] = dw2 + self.reg*self.params['W2']
       grads['b2'] = db2
        
       dx1,dw1,db1 = affine_relu_backward(dx2,ar1_cache)
       grads['W1'] = dw1 + self.reg*self.params['W1']
       grads['b1'] = db1
        
       return loss,grads
    
    


# In[ ]:




