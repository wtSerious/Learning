#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np


# In[13]:


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
    v = config['momentum'] * v - config['learning_rate']*dw
    next_w = w + v
    
    config['velocity'] = v
    
    return next_w,v


# In[14]:


get_ipython().system('jupyter nbconvert --to python optim.ipynb')


# In[ ]:




