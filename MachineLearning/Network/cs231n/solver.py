#!/usr/bin/env python
# coding: utf-8

# In[26]:


import sys;
sys.path.append("E:/JupyterEnviroment/Learning/MachineLearning/Network/cs231n/")
import optim
import numpy as np


# In[35]:


class Solver(object):
    def __init__(self,model,data,**kwargs):
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val,self.y_val = data["X_val"],data["y_val"]
        self.update_rule = kwargs.pop("update_rule","sgd_momentum")
        self.optim_config = kwargs.pop('optim_config',{})
        self.lr_decay = kwargs.pop('lr_decay',1.0)
        self.batch_size = kwargs.pop('batch_size',100)
        self.num_epochs = kwargs.pop('num_epochs',10)
        self.print_every = kwargs.pop('print_every',10)
        self.verbose = kwargs.pop('verbose',True)
       
        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ','.join('"%s"'%k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s'%extra)
         
        # Make sure the update rule exist,then replace the string
        # name with  the actual function
        if not hasattr(optim,self.update_rule):
            raise ValueError('Invalid update_rule"%s"'%self.update_rule)
        self.update_rule = getattr(optim,self.update_rule)
        self._reset()
        
    def _reset(self):
        """
        Set up some book-keeping variables for optimization.
        Don't call this manually.
        """
        #Set up some variable for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
        #Make a deep copy of the optim_config for each parameters
        self.optim_configs = {}
        for p in self.model.params:
            d = {k:v for k,v in self.optim_config.items()}
            self.optim_configs[p] = d
    
    def _step(self):
        """
        Make a single gradient update. This is called
        by train() and should not be called manually.
        """ 
        # Make a minibatch of trainning data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train,self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        
        #Compute the loss and gradient
        loss,grads = self.model.loss(X_batch,y_batch)
        self.loss_history.append(loss)
        for p,w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]#这是啥
            next_w,next_config = self.update_rule(w,dw,config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config
    
    def check_accuracy(self,X,y,num_samples = None,batch_size = 100):
        N = X.shape[0]
        if num_samples is not None and N > batch_size:
            mask = np.random.choice(N,num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
            
        #Compute predictions in batches
        num_batches = int(N/batch_size)
        if N % batch_size !=0:
            num_batches += 1

        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i+1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores,axis = 1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred==y)
        
        return acc
    
    def train(self):
        num_train = self.X_train.shape[0]
        
        #计算整个数据集有多少个 batch_size 大小
        iterations_per_epoch = max(int(num_train/self.batch_size),1)
        
        num_iterations = iterations_per_epoch *self.num_epochs
        
        for i in range(num_iterations):
            self._step()
            if self.verbose and i % self.print_every == 0:
                print ("(Iteration %d / %d) loss: %f" % (i + 1, num_iterations, self.loss_history[-1]))
            
            epoch_end = ((i+1) % iterations_per_epoch) == 0
            if epoch_end:
                self.epoch +=1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate']*=self.lr_decay
                
            first_it = (i==0)
            last_it = (i==num_iterations+1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train,self.y_train,num_samples=1000)
                val_acc = self.check_accuracy(self.X_val,self.y_val,num_samples=1000)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                if self.verbose:
                    print ('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                     self.epoch, self.num_epochs, train_acc, val_acc))
                
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_params = {}
                        for k,v in self.model.params.items():
                            self.best_params[k] = v.copy()


# In[32]:


get_ipython().system('jupyter nbconvert --to script solver.ipynb')


# In[ ]:




