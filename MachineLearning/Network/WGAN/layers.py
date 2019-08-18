import torch.nn as nn
import torch

class MinibatchDiscriminator(nn.Module):
    def __init__(self,insize,numkernel,dimsize):
        super(MinibatchDiscriminator,self).__init__()
        self.numkernel = numkernel
        self.dimsize = dimsize
        self.insize = insize
        self.layer = nn.Linear(insize,numkernel*dimsize)
    def forward(self,x):
        feature = self.layer(x).view(-1,self.numkernel,self.dimsize)
        feature = feature.unsqueeze(3)
        feature_ = feature.permute(3,1,2,0)
        feature,feature_ = torch.broadcast_tensors(feature,feature_)
        norm = torch.sum(torch.abs(feature-feature_),dim=2)
        eraser = torch.eye(feature.shape[0],device=torch.device('cuda')).view(feature.shape[0],1,feature.shape[0])
        eraser,norm = torch.broadcast_tensors(eraser,norm)
        c_b = torch.exp(-(norm+eraser*1e6))
        o_b = torch.sum(c_b,dim=2)
        x = torch.cat((x,o_b),dim=1)
        return x

class Reshape(nn.Module):
    def __init__(self, _shape):
        super(Reshape, self).__init__()
        self.shape = _shape

    def forward(self, x):
        return x.view(-1, *self.shape)