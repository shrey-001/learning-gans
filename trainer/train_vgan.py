import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import AverageMeter
 
def train_on_epoch(dataloader,generator,discriminator,g_optim,d_optim,criterion,cur_epoch,device,wandb_log=True):
    """
    Min-Max game
    """
    disc_image_accuracy = AverageMeter()
    disc_fake_accuracy = AverageMeter()

    l_D = AverageMeter()
    l_G = AverageMeter()

    for image, labels in dataloader:
        image = image.view(-1,784).to(device)
        batch_size = image.shape[0] #100

        # Train the Discriminator
        # max log(D(x))+log(1-D(G(z)))

        noise = torch.randn(batch_size,100).to(device)
        fake = generator(noise)

        disc_image = discriminator(image).view(-1)
        disc_fake = discriminator(fake).view(-1)

        disc_image_accuracy.update(disc_image.mean().item())
        disc_fake_accuracy.update(disc_fake.mean().item())
        
        loss_image_disc = criterion(disc_image, torch.ones_like(disc_image))
        loss_fake_disc = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = loss_image_disc + loss_fake_disc
        l_D.update(loss_disc.item())

        discriminator.zero_grad()
        loss_disc.backward(retain_graph = True)
        d_optim.step()


        # Train the Generator
        # min log(1-D(G(z))) <==> max log(D(G(z)))

        output = discriminator(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        l_G.update(loss_gen.item())

        generator.zero_grad()
        loss_gen.backward()
        g_optim.step()
    print("Epoch :",cur_epoch,",Loss Discriminator","%.4f"%(l_D.avg),",Loss Generator","%.4f"%(l_G.avg))
    if wandb_log:
        wandb.log({"disc_image_accuracy":disc_image_accuracy.avg,
                    "disc_fake_accuracy":disc_fake_accuracy.avg,
                    "Loss Discriminator":l_D.avg,
                    "Loss Generator":l_G.avg},
                    step = cur_epoch)


    
