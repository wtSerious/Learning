#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


# In[17]:


def affine_forward(x,w,b):
    
    """
Compute the forward pass for affine layer

Parameters
----------
x: A numpy array containing input data,of shape(N,d_1,d_2,..,d_k) 
w: A numpy array of weights, of shape (D,M)
b: A numpy array of biases, of shape (M,)

Return a tuple of
----------
out: output, of shape(N,M)
cache: (x,w,b)
    """
    N = x.shape[0]
    x_rsp = x.reshape(N,-1)
    out = x_rsp.dot(w) + b
    cache = (x,w,b)
    return out,cache


# In[41]:


def relu_forward(x):
    """
    Compute the forward pass for a layer of relu 
    """
    out = x*(x >= 0)
    cache = x
    return out,cache


# In[42]:


def affine_backward(dout,cache):
    """
    Compute the backward  pass for an affine layers.py
    
    Parameters:
    -----------
    dout: Upstream derivative of shape(N,M)
    cache: Tuple of
        x: Input data
        w: Weight, of shape(D,M)
        
    Return:
    -----------
    dx: Gradient with respect to x, of shape(N,d_1,d_2,...,d_k)
    db: Gradient with respect to b, of shape(M,)
    """
    x,w,b = cache
    x_rsp = x.reshape(x.shape[0],-1)
    dx = dout.dot(w.T)
    dx = dx.reshape(*x.shape)
    dw = x_rsp.T.dot(dout)
    db = np.sum(dout,axis= 0)
    
    return dx,dw,db


# In[39]:


def relu_backward(dout,cache):
    dx = (cache>=0 ) * dout
    return dx


# In[55]:


def affine_relu_forward(x,w,b):
    """
    Convenience layer that perorms an affine transform 
    followed by Relu
    
    Parameter:
    ----------
    x: A numpy array of input data, of shape (N,M)
    w: A numpy array of weight
    
    Return a typle of:
    ----------
    out: Output from the Relu
    cache: Object to give to the backward pass
    """
    z,af_cache = affine_forward(x,w,b)
    out,relu_cahce = relu_forward(z)
    cache = (af_cache,relu_cahce)
    return out,cache


# In[57]:


def affine_relu_backward(dout,cache):
    af_cache,relu_cache = cache
    dr = relu_backward(dout,relu_cache)
    dx,dw,db = affine_backward(dr,af_cache)
    return dx,dw,db


# In[60]:


def soft_loss(x,y):
    """
    
    """
    probs = np.exp(x-np.max(x,axis=1,keepdims=True))
    probs /= np.sum(probs,axis=1,keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log([probs[np.arange(N),y]]))/N
    dx = probs.copy()
    dx[np.arange(N),y] -= 1
    dx /= N
    return loss, dx


# In[58]:


get_ipython().system('jupyter nbconvert --to python layers.ipynb')

