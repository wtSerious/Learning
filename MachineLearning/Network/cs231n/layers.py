#!/usr/bin/env python
# coding: utf-8

# In[85]:


import sys;
sys.path.append("E:/JupyterEnviroment/Learning/MachineLearning/Network/cs231n")
import numpy as np
from fast_layers import *


# In[77]:


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


# In[78]:


def relu_forward(x):
    """
    Compute the forward pass for a layer of relu 
    """
    out = x*(x >= 0)
    cache = x
    return out,cache


# In[79]:


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    
    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x,axis = 0)
        sample_var = np.var(x,axis = 0)
        x_hat = (x-sample_mean)/(np.sqrt(sample_var+eps))
        out = gamma * x_hat + beta
        cache = (gamma,x,sample_mean,sample_var,eps,x_hat)
        running_mean = momentum * running_mean + (1-momentum)*sample_mean
        running_var = momentum * running_var + (1-momentum)*sample_var
    elif mode == 'test':
        scale = gamma / (np.sqrt(running_var  + eps))
        out = x * scale + (beta - running_mean * scale)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
        
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


# In[80]:


def dropout_forward(x,dropout_param):
    p,mode = dropout_param['p'],dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    mask = None
    out = None
    
    if mode == 'train':
        #在这里采用了在训练的时候就扩大了输出，因此在测试时不用
        #缩小输出
        mask = (np.random.rand(*x.shape)>=p)/(1-p)
        out = x * mask
    
    elif mode == 'test':
        out = x
    cache = (dropout_param,mask)
    out = out.astype(x.dtype,copy = False)
    
    return out, cache


# In[6]:


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


# In[7]:


def affine_bn_relu_forward(x,w,b,gamma,beta,bn_param):
    a,fc_cache = affine_forward(x,w,b)
    bn,bn_cache = batchnorm_forward(a,gamma,beta,bn_param)
    out,relu_cache = relu_forward(bn)
    cache = (fc_cache,bn_cache,relu_cache)
    return out, cache


# In[8]:


def conv_forward_naive(x,w,b,conv_params):
    """
    A naive implementation of the forward pass for a convolutional layer.
    
    The input consists of N data point ,each with C channels, heights H and width
    W. We convolve each input with F different filters,where each filter 
    spans all C channels and has height HH and width HH.
    
    Parameters:
    -----------
    - x: Input data of shape (N,C,H,W)
    - w: Filter weights of shape (F,C,HH,WW)
    - b: Biases,of shape(F,)
    - conv_params: A dictionary with the follow keys:
     - 'stride': The number of pixels between adjacent receptive fields in the
     horizontal and vertical directions.
     - 'pad': The number of pixels that will be used to zero-pad the input.
     
     Return a tuple of:
     ------------------
     - out: Output data ,of shape(N,F,H',W') where H' and W' were given by
       H' = 1 + (H+2*pad-HH)/stride
       W' = 1 + (W+2*pad-HH)/stride
     - cache:(x,w,b,conv_param)
   """
    N,C,H,W = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
    F,HH,WW = w.shape[0],w.shape[2],w.shape[3]
    stride,pad = conv_params['stride'],conv_params['pad']
    data = np.pad(x,((0,),(0,),(pad,),(pad,)),mode='constant',constant_values = 0)
    _H = int((H + 2 * pad - HH) / stride) + 1
    _W = int((W + 2 * pad - WW) / stride) + 1
    out = np.zeros((N,F,_H,_W))
    
    for i in range(_H):
        for j in range(_W):
            x_mask = data[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]
            for k in range(F):
                out[:,k,i,j] = np.sum(x_mask*w[k,:,:,:],axis=(1,2,3))
                
    out = out + (b)[None,:,None,None]
    cache = (x,w,b,conv_params)
    return out,cache


# In[9]:


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    N,C,H,W = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
    HH,WW,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
    _H = int((H-HH)/stride)+1
    _W = int((W-WW)/stride)+1
    out = np.zeros((N,C,_H,_W))
    
    for i in range(_H):
        for j in range(_W):
            x_mask = x[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]
            out[:,:,i,j] = np.max(x_mask, axis=(2,3)) 
    
    cache = (x,pool_param)
    return out,cache
    


# In[ ]:


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
  
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


# In[88]:


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    con_out,con_cache = conv_forward_fast(x=x,w=w,b=b,conv_param=conv_param)
    r_out,r_cache = relu_forward(con_out)
    out ,pool_cache = max_pool_forward_fast(r_out,pool_param)
    cache = (con_cache,r_cache,pool_cache)
    return out,cache
    


# In[10]:


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x,pool_param = cache
    N,C,H,W = x.shape
    HH,WW,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
    _H = int((H-HH)/stride)+1
    _W = int((W-WW)/stride)+1
    dx = np.zeros_like(x)
    
    for i in range(_H):
        for j in range(_W):
            x_mask = x[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]
            x_mask_max = np.max(x_mask,axis=(2,3))
            temp_binary_mask = (x_mask==(x_mask_max)[:,:,None,None])
            dx[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW] += (dout[:,:,i,j])[:,:,None,None]*temp_binary_mask
    return dx


# In[11]:


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


# In[12]:


def relu_backward(dout,cache):
    dx = (cache>=0 ) * dout
    return dx


# In[13]:


def batchnorm_backward(dout,cache):
    dx, dgamma, dbeta = None, None, None
    gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
    N = x.shape[0]

    dx_1 = gamma * dout
    dx_2_b = np.sum((x - u_b) * dx_1, axis=0)
    dx_2_a = ((sigma_squared_b + eps) ** -0.5) * dx_1
    dx_3_b = (-0.5) * ((sigma_squared_b + eps) ** -1.5) * dx_2_b
    dx_4_b = dx_3_b * 1
    dx_5_b = np.ones_like(x) / N * dx_4_b
    dx_6_b = 2 * (x - u_b) * dx_5_b
    dx_7_a = dx_6_b * 1 + dx_2_a * 1
    dx_7_b = dx_6_b * 1 + dx_2_a * 1
    dx_8_b = -1 * np.sum(dx_7_b, axis=0)
    dx_9_b = np.ones_like(x) / N * dx_8_b
    dx_10 = dx_9_b + dx_7_a

    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = dx_10
    return dx, dgamma, dbeta
    


# In[14]:


def dropout_backward(dout,cache):
    dropout_param,mask = cache
    mode = dropout_param['mode']
    
    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx    
    


# In[15]:


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x,w,b,conv_params = cache
    N,C,H,W = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
    F,HH,WW = w.shape[0],w.shape[1],w.shape[2]
    stride,pad = conv_params['stride'],conv_params['pad']
    _H = int((H+2*pad-HH)/stride) + 1
    _W = int((H+2*pad-HH)/stride) + 1
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    db = np.sum(dout, axis = (0,2,3))
    for i in range(_H):
        for j in range(_W):
            x_mask = x_pad[:,:,i*stride:(i*stride)+HH,j*stride:(j*stride)+WW]
            for k in range(F):
                #d[i,j] = dout[i,j] * x_mask[:,:]
                dw[k,:,:,:] += np.sum(x_mask[:,:,:,:]*(dout[:, k, i, j])[:,None,None,None],axis=0)
            for n in range(N):
                
                dx_pad[n,:,i*stride:(i*stride)+HH,j*stride:(j*stride)+WW] += np.sum(
                    (dout[n,:,i,j])[:,None,None,None]*w[:,:,:,:],axis=0)
    
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    return dx, dw, db


# In[ ]:


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


# In[16]:


def affine_relu_backward(dout,cache):
    af_cache,relu_cache = cache
    dr = relu_backward(dout,relu_cache)
    dx,dw,db = affine_backward(dr,af_cache)
    return dx,dw,db


# In[17]:


def affine_bn_relu_backward(dout, cache):
    fc_cache,bn_cache,relu_cache = cache
    dbn = relu_backward(dout,relu_cache)
    da,dgammg,dbeta = batchnorm_backward(dbn,bn_cache)
    dx,dw,db = affine_backward(da,fc_cache)
    return dx,dw,db,dgammg,dbeta


# In[87]:


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    con_cache,r_cache,pool_cache = cache
    pool_back_out = max_pool_backward_fast(dout,pool_cache)
    r_back_out = relu_backward(pool_back_out,r_cache)
    dx,dw,db = conv_backward_fast(r_back_out,con_cache)
    return dx,dw,db


# In[19]:


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


# In[91]:


get_ipython().system('jupyter nbconvert --to python layers.ipynb')

