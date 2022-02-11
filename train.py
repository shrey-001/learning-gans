import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import models.vanilla_gan as vgan
from data.dataloader import get_dataloader
import trainer.train_vgan as vgan_train 
from eval import evaluate

wandb.init(project="GANs", entity="shreyanshsaxena")

device = "cuda"
lr = 3e-4
num_of_epoch = 50

batch_size = 100
dat = "MNIST"
num_workers = 4
optimizer = "Adam"
fixed_noise = torch.randn((32, 100)).to(device)  

dataloader = get_dataloader(dat,batch_size,num_workers)

generator = vgan.Generator(n_feature=100,n_out=784).to(device)
discriminator = vgan.Discriminator(n_feature=784).to(device) 

g_optim = optim.Adam(generator.parameters(), lr=lr)
d_optim = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(num_of_epoch):
    vgan_train.train_on_epoch(dataloader,generator,discriminator,g_optim,d_optim,criterion,epoch,device)
    evaluate(generator, fixed_noise,epoch)