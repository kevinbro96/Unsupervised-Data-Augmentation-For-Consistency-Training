from __future__ import print_function
import argparse
import random # to set the python random seed

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import torch.optim as optim
from torchvision import datasets, transforms
# Ignore excessive warnings
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# WandB – Import the wandb library
import wandb
wandb.login()
wandb.init(project="dcgan") # Change the project name based on your W & B account

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 32

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 30

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def train(args, gen, disc, device, dataloader, optimizerG, optimizerD, criterion, epoch, iters):
  gen.train()
  disc.train()
  img_list = []
  fixed_noise = torch.randn(64, config.nz, 1, 1, device=device)

  # Establish convention for real and fake labels during training (with label smoothing)
  real_label = 0.9
  fake_label = 0.1
  for i, data in enumerate(dataloader, 0):

      #*****
      # Update Discriminator
      #*****
      ## Train with all-real batch
      disc.zero_grad()
      # Format batch
      real_cpu = data[0].to(device)
      b_size = real_cpu.size(0)
      label = torch.full((b_size,), real_label, device=device)
      # Forward pass real batch through D
      output = disc(real_cpu).view(-1)
      # Calculate loss on all-real batch
      errD_real = criterion(output, label)
      # Calculate gradients for D in backward pass
      errD_real.backward()
      D_x = output.mean().item()

      ## Train with all-fake batch
      # Generate batch of latent vectors
      noise = torch.randn(b_size, config.nz, 1, 1, device=device)
      # Generate fake image batch with G
      fake = gen(noise)
      label.fill_(fake_label)
      # Classify all fake batch with D
      output = disc(fake.detach()).view(-1)
      # Calculate D's loss on the all-fake batch
      errD_fake = criterion(output, label)
      # Calculate the gradients for this batch
      errD_fake.backward()
      D_G_z1 = output.mean().item()
      # Add the gradients from the all-real and all-fake batches
      errD = errD_real + errD_fake
      # Update D
      optimizerD.step()

      #*****
      # Update Generator
      #*****
      gen.zero_grad()
      label.fill_(real_label)  # fake labels are real for generator cost
      # Since we just updated D, perform another forward pass of all-fake batch through D
      output = disc(fake).view(-1)
      # Calculate G's loss based on this output
      errG = criterion(output, label)
      # Calculate gradients for G
      errG.backward()
      D_G_z2 = output.mean().item()
      # Update G
      optimizerG.step()

      # Output training stats
      if i % 50 == 0:
          print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, args.epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
          wandb.log({
              "Gen Loss": errG.item(),
              "Disc Loss": errD.item()})

      # Check how the generator is doing by saving G's output on fixed_noise
      if (iters % 500 == 0) or ((epoch == args.epochs-1) and (i == len(dataloader)-1)):
          with torch.no_grad():
              fake = gen(fixed_noise).detach().cpu()
          img_list.append(wandb.Image(vutils.make_grid(fake, padding=2, normalize=True)))
          wandb.log({
              "Generated Images": img_list})
      iters += 1


wandb.watch_called = False
# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config  # Initialize config
config.batch_size = batch_size
config.epochs = num_epochs
config.lr = lr
config.beta1 = beta1
config.nz = nz
config.no_cuda = False
config.seed = manualSeed  # random seed (default: 42)
config.log_interval = 10  # how many batches to wait before logging training status


def main():
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(config.seed)  # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    np.random.seed(config.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Load the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                              shuffle=True, num_workers=workers)

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    wandb.watch(netG, log="all")
    wandb.watch(netD, log="all")
    iters = 0
    for epoch in range(1, config.epochs + 1):
        train(config, netG, netD, device, trainloader, optimizerG, optimizerD, criterion, epoch, iters)

    # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
    torch.save(netG.state_dict(), "model.h5")
    wandb.save('model.h5')


if __name__ == '__main__':
    main()
