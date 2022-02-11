import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self,n_feature):
    super(Discriminator,self).__init__()
    self.n_feature = n_feature   #  28*28
    self.disc = nn.Sequential(
        nn.Linear(self.n_feature, 128),
        nn.LeakyReLU(0.1),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
  def forward(self, x):
    return self.disc(x)

class Generator(nn.Module):
  def __init__(self,n_feature,n_out):
    super(Generator,self).__init__()
    self.n_feature = n_feature
    self.n_out = n_out  

    self.gen = nn.Sequential(
        nn.Linear(self.n_feature, 256),
        nn.LeakyReLU(0.1),
        nn.Linear(256, self.n_out),
        nn.Tanh()
    )

  def forward(self, x):
    return self.gen(x)