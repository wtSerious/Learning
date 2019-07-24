#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np


# In[4]:


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
    
    return next_w,config


# In[2]:


def sgd(w, dw, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)

  w -= config['learning_rate'] * dw
  return w, config


# In[ ]:


def adam(x, dx, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)
  
  next_x = None
  #############################################################################
  # TODO: Implement the Adam update formula, storing the next value of x in   #
  # the next_x variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                         #
  #############################################################################
  config['t'] += 1
  config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
  config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx**2)
  mb = config['m'] / (1 - config['beta1']**config['t'])
  vb = config['v'] / (1 - config['beta2']**config['t'])
  next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return next_x, config


# In[4]:


get_ipython().system('jupyter nbconvert --to python optim.ipynb')


# In[ ]:




