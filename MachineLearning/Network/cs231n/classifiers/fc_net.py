#!/usr/bin/env python
# coding: utf-8

# In[19]:


import sys;
sys.path.append("E:/JupyterEnviroment/Learning/MachineLearning/Network/")
import numpy as np
from cs231n.layers import *


# In[9]:


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
    
    


# In[22]:


class FullConnetctedNet(object):
    """
     A full-connetcted neural network with an arbitrary numer
    of hidden layers, Relu nonlinearities and a softmax loss funciton.
     This will also implement dropout and batch normalization as options.
    For L layers , the architecture will be 
     {affine - [batch norm] - relu - [dropout]} x (L-1) - affine - 
    softmax where batch normalization and dropout are optional,and
    the {...} block is repeated L-1 times.
    
    Similar to the TwoLayerNet above, learnalble parameters are 
    stored in the self.params dictionary and will be learned using 
    the Solver class.
    """
    def __init__(self,hidden_dims,input_dim = 3*32*32,num_classes = 10,
                dropout = 0,use_batchnorm = False,reg = 0.0,weight_scale = 1e-2,
                dtype = np.float32,seed = None):
        """
        Initialize a new FullyConnetedNet.
        
        Parameters:
        -----------
        - hidden_dims: A list of integers giving the size of 
        each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes:An integer giving the number of classes to classify.
        - dropout: Scaler between 0 and 1 giving dropout strength. If
        dropout = 0 then the network should not use dropout at all.
        - use_batchnorm: Whether or not use the network should use
        normalization.
        - reg: Scaler giving L2 regularization strength.
        - weight_scale: Scaler giving the standard deviation for random
        initialization of the weights.
        - dtype: A numpy datatype object; all computations will be 
        preformed using this datatype. float32 is faster but less accurate.
        use float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout l
        layers. This will make the dropout layers deteriminstic so we can 
        gradient the model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        #initialize weights and b
        layer_input_dim = input_dim
        for i,h in enumerate(hidden_dims):
            self.params['W%d'%(i+1)] = weight_scale * np.random.randn(layer_input_dim,h) 
            self.params['b%d'%(i+1)] = weight_scale * np.zeros(h)
            if self.use_batchnorm:
                self.params['gamma%d'%(i+1)] = np.ones(h)
                self.params['beta%d'%(i+1)] = np.zeros(h)
            layer_input_dim = h
        self.params['W%d'%(self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
        self.params['b%d'%(self.num_layers)] = weight_scale * np.zeros(num_classes)
        
        #dropout
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train','p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'model': 'train'} for i in range(self.num_layers-1)]
        
        for k,v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    def loss(self,X,y = None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.bn_params is not None:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
            
        scores = None
        
        layer_input = X
        ar_cache = {}
        dp_cache = {}
        
        for lay in range(self.num_layers-1):
            if self.use_batchnorm:
                layer_input,ar_cache[lay] = affine_bn_relu_forward(
                layer_input,self.params['W%d'%(lay+1)],
                self.params['b%d'%(lay+1)],self.params['gamma%d'%(lay+1)],
                self.params['beta%d'%(lay+1)],self.bn_params[lay])
            else:
                layer_input, ar_cache[lay] = affine_relu_forward(x=layer_input,w=self.params['W%d'%(lay+1)]
                                                                 ,b=self.params['b%d'%(lay+1)])
            if self.use_dropout:
                layer_input,dp_cache[lay] = dropout_forward(x=layer_input,dropout_param=self.dropout_param)
        
        #calculate the output of the last layer(output layer)      
        arout,ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d'%(self.num_layers)], self.params['b%d'%(self.num_layers)])
        
        scores = arout
        
        if mode =='test':
            return scores
        
        loss,grads = 0.0,{}
        
        loss ,dscores = soft_loss(scores,y)
        dhout = dscores
        loss = loss + 0.5 * self.reg * np.sum(self.params['W%d'%(self.num_layers)] * self.params['W%d'%(self.num_layers)])
        
        #calculate the gradient of the output layer first
        dx , dw , db = affine_backward(dhout , ar_cache[self.num_layers])
        grads['W%d'%(self.num_layers)] = dw + self.reg * self.params['W%d'%(self.num_layers)]
        grads['b%d'%(self.num_layers)] = db
        dhout = dx
        
        #calculate the gradient of the hidden layers then
        for idx in range(self.num_layers-1):
            lay = self.num_layers - 1 - idx - 1
            loss = loss = loss + 0.5 * self.reg * np.sum(self.params['W%d'%(lay+1)] * self.params['W%d'%(lay+1)])
            if self.use_dropout:
                dx = dropout_backward(dx,dp_cache[lay])
            if self.use_batchnorm:
                dx,dw,db,dgama,dbeta = affine_bn_relu_backward(dx,ar_cache[lay])
            else:
                dx,dw,db = affine_relu_backward(dx,ar_cache[lay])
            
            grads['W%d'%(lay+1)] = dw + self.reg*self.params['W%d'%(lay+1)]
            grads['b%d'%(lay+1)] = db 
            if self.use_batchnorm:
                grads['gamma%d'%(lay+1)] = dgama
                grads['beta%d'%(lay+1)] = dbeta
            dhout = dx
            
        return loss,grads
        
        


# In[8]:


get_ipython().system('jupyter nbconvert --to python fc_net.ipynb')


# In[ ]:




