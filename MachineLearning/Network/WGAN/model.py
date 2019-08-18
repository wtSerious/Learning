import torch.nn as nn
import torch
from layers import MinibatchDiscriminator
from layers import Reshape

class Generator(nn.Module):
    def __init__(self, indim):
        super(Generator, self).__init__()

        def block(indim, outdim, normalize=True):
            layers = [nn.Linear(indim, outdim)]
            if normalize:
                layers.append(nn.BatchNorm1d(outdim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
                                   # nn.Linear(100, 512 * 6 * 6),
                                   # nn.ReLU(inplace=True),
                                   # nn.BatchNorm1d(512 * 6 * 6),
                                   # Reshape((512, 6, 6)),
                                               nn.ConvTranspose2d(100,512,kernel_size=6,stride=1,padding=0,bias=False),
                                                                      nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.Tanh()
                                   )

    def forward(self, x):
        out = self.model(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(512,1,kernel_size=6,stride = 1,padding=0,bias=False)
                                   )
        # self.mnibt_dismtor = MinibatchDiscriminator(insize=18432, numkernel=50, dimsize=10)
        # self.lastlayer = nn.Linear(18432 + 50, 1)

    def forward(self, x):
        out = self.model(x)
        # x = x.view(x.shape[0], -1)
        # feature = self.mnibt_dismtor(x)
        # out = self.lastlayer(feature)
        # if matching:
        #     return feature, out
        return out.mean()