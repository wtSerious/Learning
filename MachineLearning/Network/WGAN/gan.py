import torch
from torch.utils.tensorboard import SummaryWriter
import Load_faces as dst
# import torchvision.datasets as dst
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import optim
from model import *
import torch.backends.cudnn as cudnn
import torchvision
import os
import random
from dcgan import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Gan:
    def __init__(self,batch_size,indim,c):

        self.device = torch.device('cuda')
        self.writer = SummaryWriter(log_dir='/home/wt/mine/model_tensorboard/wgan5/')
        self.model_path = "/home/wt/mine/model_data/wgan5/"
        self.indim = indim
        self.model_g = Generator(100)
        self.model_g.apply(weights_init)
        self.model_d = Discriminator()
        self.model_d.apply(weights_init)
        self.model_d.cuda()
        self.model_g.cuda()
        self.batch_size = batch_size
        self.eval_data = torch.randn(batch_size, indim,1,1, device=self.device)
        self.c = c

    def load_data(self,data_path):
        transform = T.Compose([T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
        train_data = dst.load_Face(data_path,number=6400,transform=transform)
        # train_data = dst.ImageFolder(data_path,transform=transform)
        self.data = DataLoader(train_data,batch_size=self.batch_size,shuffle=True)


    def train(self, load_model=False, path=None, epoch_low=0, epoch_high=100, dis=10):
        manualSeed = random.randint(1, 10000)  # fix seed
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        noise = torch.FloatTensor(32, 100, 1, 1)
        fixed_noise = torch.FloatTensor(32, 100, 1, 1).normal_(0, 1)
        one = torch.FloatTensor([1])
        mone = one * -1

        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        self.optimizerD = optim.RMSprop(self.model_d.parameters(), lr=0.00005)
        self.optimizerG = optim.RMSprop(self.model_g.parameters(), lr=0.00005)

        for epoch in range(epoch_low, epoch_high):
            data_iter = iter(self.data)
            i = 0
            while i < len(self.data):
                ############################
                # (1) Update D network
                ###########################
                for p in self.model_d.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                # train the discriminator Diters times
                j = 0
                while j < 1 and i < len(self.data):
                    j += 1
                    #!!!!!!!!!!!!!!!!!!!!!!!问题就出在更新这里！！！！！！！！！！！！！！！！！！！！！！！

                    ########TRUEEE
                    #clamp parameters to a cube
                    for p in self.model_d.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    data, y = data_iter.next()
                    i += 1

                    self.model_d.zero_grad()
                    inputv = data.cuda()
                    errD_real = self.model_d(inputv)
                    errD_real.backward(one)

                    # train with fake
                    noise.resize_(32, 100, 1, 1).normal_(0, 1)
                    fake = self.model_g(noise.detach())  # totally freeze netG

                    inputv = fake
                    errD_fake = self.model_d(inputv)
                    errD_fake.backward(mone)
                    errD = errD_real - errD_fake
                    self.optimizerD.step()

                # (2) Update G network
                ###########################
                for p in self.model_d.parameters():
                    p.requires_grad = False  # to avoid computation
                self.model_g.zero_grad()
                # in case our last batch was the tail batch of the dataloader,
                # make sure we feed a full batch of noise
                noise.resize_(32, 100, 1, 1).normal_(0, 1)

                fake = self.model_g(noise.detach())
                errG = self.model_d(fake)
                errG.backward(one)
                self.optimizerG.step()
            self.save_loss(errG,-errG,errD,-errD,errD_real,errD_fake,epoch)

            if epoch % dis == 0:
                self.eval(epoch)

    def eval(self,step=0):
        self.model_g.eval()
        fake_img = self.model_g(self.eval_data)
        self.save_picture(fake_img,step)
        path = os.path.join(self.model_path,'model_epoch{}'.format(step))
        self.save_model_dict(path)
        self.model_g.train()

    def save_model_dict(self,path):
        checkpoint = {'model_g': self.model_g.state_dict(),'model_d ':self.model_d.state_dict(),
                      'optimizer_g':self.optimizerG.state_dict(),'optimizer_d': self.optimizerD.state_dict()}
        torch.save(checkpoint, path)

    def load_model_dict(self, path):
        checkpoint = torch.load(path)
        # print(checkpoint.keys())
        self.model_g.load_state_dict(checkpoint['model_g'])
        # print(checkpoint['model_d'])
        self.model_d.load_state_dict(checkpoint['model_d '])
        # self.optimizer_d = optim.Adam(self.model_d.parameters(), lr=1e-4, betas=(0.5, 0.9))
        # self.optimizer_g = optim.Adam(self.model_g.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optimizerG = optim.RMSprop(self.model_g.parameters(),lr=1e-4)
        self.optimizerD = optim.RMSprop(self.model_d.parameters(),lr=1e-4)
        self.optimizerG .load_state_dict(checkpoint['optimizer_g'])
        self.optimizerD.load_state_dict(checkpoint['optimizer_d'])

    # def weights_init(self,m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         m.weight.data.normal_(0.0, 0.02)
    #     elif classname.find('BatchNorm') != -1:
    #         m.weight.data.normal_(1.0, 0.02)
    #         m.bias.data.fill_(0)



    def save_loss(self,loss_g,g_cost,loss_d,wass_d,loss_d_real,loss_d_fake,step):
        self.writer.add_scalars('loss', {'gloss_g': loss_g, 'g_cost':g_cost,'gloss_d': loss_d,'wass_d':wass_d, 'gloss_d_real': loss_d_real,
                                    'gloss_d_fake': loss_d_fake}, global_step=step)

    def save_picture(self,fake_img,step):
        store_img = torchvision.utils.make_grid((fake_img + 1) / 2.0, padding=2)
        group = step//100
        self.writer.add_image('generate_img{}'.format(group), store_img, global_step=step)
        self.writer.close()