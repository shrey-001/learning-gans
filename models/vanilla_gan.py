import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    n_features = 784   #  28*28
    self.disc = nn.Sequential(
        nn.Linear(n_features, 128),
        nn.LeakyReLU(0.1),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
  def forward(self, x):
    return self.disc(x)

class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    n_feature = 100
    n_out = 784  

    self.gen = nn.Sequential(
        nn.Linear(n_feature, 256),
        nn.LeakyReLU(0.1),
        nn.Linear(256, n_out),
        nn.Tanh()
    )

  def forward(self, x):
    return self.gen(x)