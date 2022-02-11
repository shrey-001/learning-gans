import torch
import torchvision
import wandb

def evaluate(generator, fixed_noise,cur_epoch,wandb_log=True,reshape=True):
    generator.eval()
    with torch.no_grad():
        if reshape:
            fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
        else:
            fake = generator(fixed_noise)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        images = wandb.Image(img_grid_fake, caption="Generated Images")
        if wandb_log:
            wandb.log({'Generated Examples': images}
                        ,step = cur_epoch)