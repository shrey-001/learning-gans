import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self,nc,ndf):
    super(Discriminator,self).__init__()
    self.nc = nc
    self.ndf = ndf
    self.disc = nn.Sequential(
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        nn.Sigmoid()
    )
  def forward(self, x):
    return self.disc(x)
class Generator(nn.Module):
  def __init__(self,nc,nz,ngf):
    super(Generator,self).__init__()
    self.nc = nc
    self.nz = nz
    self.ngf = ngf
    self.gen = nn.Sequential(
        nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False,),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        nn.ConvTranspose2d( ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
        nn.Tanh()
    )

  def forward(self, x):
    return self.gen(x)