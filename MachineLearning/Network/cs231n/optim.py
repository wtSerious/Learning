#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


def sgd_momentum(w,dw,config=None):
    """
    Performs stochastic gradient descent with momentum.
    
    Parameters:
    -----------
    config format:
        learning_rate:
        momentum:
        velocity:
    """
    if config is None:
        config = {}
    
    config.setdefault('learning_rate',1e-2)
    config.setdefault('momentum',0.9)
    v = config.get('velocity',np.zeros_like(w))
    
    next_w = None
    v = config['momentum'] * v - config['learning_tate']*dw
    next_w = w + v
    
    config['velocity'] = v
    
    return next_w,v


# In[ ]:




