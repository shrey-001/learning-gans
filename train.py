import torch

import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import torchvision
from torchvision import transforms, datasets
import wandb

import models.vanilla_gan as vgan

wandb.login()
wandb.init(project="GANs", entity="shreyanshsaxena")

lr = 3e-4
num_of_epoch = 50
batch_size = 100

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                ])
train = datasets.MNIST(
    root = './data/',
    train = True, 
    download = True, 
    transform = transform
    )
dataloader = DataLoader(train,sampler=RandomSampler(train),batch_size=batch_size,num_workers=4,drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

generator = vgan.Generator().to(device)
discriminator = vgan.Discriminator().to(device) 

g_optim = optim.Adam(generator.parameters(), lr=lr)
d_optim = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(num_of_epoch):
  for idx, (image, _) in enumerate(dataloader):
    image = image.view(-1,784).to(device)
    batch_size = image.shape[0] #100

    # Train the Discriminator
    # max log(D(x))+log(1-D(G(z)))

    noise = torch.randn(batch_size,100).to(device)
    fake = generator(noise)

    disc_image = discriminator(image).view(-1)
    disc_fake = discriminator(fake).view(-1)
    
    loss_image_disc = criterion(disc_image, torch.ones_like(disc_image))
    loss_fake_disc = criterion(disc_fake, torch.zeros_like(disc_fake))

    loss_disc = loss_image_disc + loss_fake_disc

    discriminator.zero_grad()
    loss_disc.backward(retain_graph = True)
    d_optim.step()


    # Train the Generator
    # min log(1-D(G(z))) <==> max log(D(G(z)))

    output = discriminator(fake).view(-1)
    loss_gen = criterion(output, torch.ones_like(output))

    generator.zero_grad()
    loss_gen.backward()
    g_optim.step()
    with torch.no_grad():
        if idx == 0:
                print(
                    f"Epoch [{epoch}/{num_of_epoch}] \
                        Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
                )

                acc_dis_real = disc_image.mean().item()
                acc_dis_fake = disc_fake.mean().item()

                fixed_noise = torch.randn((32, 100)).to(device)
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

                images = wandb.Image(img_grid_fake, caption="Generated Images")
                wandb.log({'Generated Examples': images,
                        'Loss Discriminator': loss_disc,
                        'Loss Generator': loss_gen,
                        'Discriminator Accuracy for Real': acc_dis_real,
                        'Discriminator Accuracy for Generated image': acc_dis_fake})

if __name__=="__main__":
    
    # Argument parser

    # Dataloader loading
    
    # Training
