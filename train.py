import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from data.dataloader import get_dataloader

import models.vanilla_gan as vgan
import models.dcgan as dcgan

import trainer.train_vgan as vgan_train 
import trainer.train_dcgan as dcgan_train
from eval import evaluate

wandb.init(project="GANs", entity="shreyanshsaxena",name ="DCGAN")

device = "cuda"
model_name = "dcgan"
lr = 3e-4
num_of_epoch = 50
nz = 100    # Latent Vector
nc = 1     # Number of input channel 
ngf = 64    # Feature Map in Generator
ndf = 64    # Feature Map in Discriminator
batch_size = 100
dat = "MNIST"
num_workers = 4
optimizer = "Adam"
fixed_noise = torch.randn((32, nz,1,1)).to(device)
generator_input = { "vgan": (nz,784),
                    "dcgan":(nc,nz,ngf)}
discriminator_input = { "vgan": (784),
                    "dcgan":(nc,ndf)} 
modelss = {"dcgan": dcgan,
        "vgan": vgan}

dataloader = get_dataloader(dat,batch_size,num_workers)

generator = modelss[model_name].Generator(nc,nz,ngf).to(device)
discriminator = modelss[model_name].Discriminator(nc,ndf).to(device) 

g_optim = optim.Adam(generator.parameters(), lr=lr)
d_optim = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(num_of_epoch):
    dcgan_train.train_on_epoch(dataloader,generator,discriminator,g_optim,d_optim,criterion,epoch,device)
    evaluate(generator, fixed_noise,epoch,reshape=False)