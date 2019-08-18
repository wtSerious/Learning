from gan import *
gan = Gan(32,100,0.01)
gan.load_data('/home/wt/mine/dataBase/faces/')
gan.train(load_model=False,path='/home/wt/mine/model_data/wgan3/model_epoch200',epoch_low=0,epoch_high=1001,dis=20)