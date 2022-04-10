## Libraries

from __future__ import print_function
import sys
import pathlib
working_directory = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(working_directory))
#sys.path.append("/gpfs/users/baillyv/dl_unmasking/utils")

#%matplotlib inline
import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.img_dataset import ImgDataset
from utils.augmentation import get__augmentation
from utils.dataset_check import dataset_check
from utils.load_config import load_config

dataset_check()
#from IPython.display import HTML


## Settings

step = 0
size_z = 100

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed) 
torch.manual_seed(manualSeed)
config = load_config()

# Data directory
dataroot = config["dataroot"]

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 16

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = size_z

# Size of feature maps in generator
ngf = 512

# Size of feature maps in discriminator
ndf = 512

# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
lr = 2.5e-5

# Beta1 hyperparam for Adam optimizers
beta1 = 0.6

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


## Getting images

unmasked_img_dir = os.path.join(dataroot, 'unmasked')
masked_img_dir = os.path.join(dataroot, 'masked')

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = ImgDataset(masked_dir=masked_img_dir, unmasked_dir=unmasked_img_dir, augmentation=get__augmentation())
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                         shuffle = True, num_workers = workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize = (8,8))
plt.axis("off")
plt.title("Training Images")
plt.imsave(f"final_results/cGAN/training_pic_{size_z}.png", np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding = 2, normalize = True).cpu(),(1,2,0)).numpy())

## Defining networks

t0 = time.time()


#### Def init

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_from_GAN(m, state_dict):
    own_state = m.state_dict()
    for name, param in state_dict.items():
        if name == 'main.0.weight':
            continue
        own_state[name].copy_(param)

#### Def generator

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz+nc, ngf * 16, 2, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 2 x 2
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, truth):
        input = torch.cat([input, truth], 1)
        return self.main(input)

# Create the generator
if step == 0:
    netG = Generator(ngpu).to(device)
    load_from_GAN(netG, torch.load("final_results/GAN/gen_2.pt"))
else:
    netG = Generator(ngpu).to(device)
    netG.load_state_dict(torch.load(f"final_results/cGAN/gen{size_z}_{step-1}.pt"))


# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
if step == 0:
    netG.apply(weights_init)

# Print the model
print(netG)


#### Def discriminator

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(2*nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 16, 1, 2, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input, truth):
        input = torch.cat([input, truth], 1)
        return self.main(input)

# Create the Discriminator
if step == 0:
    netD = Discriminator(ngpu).to(device)
    load_from_GAN(netD, torch.load("final_results/GAN/disc_3.pt"))
else:
    netD = Discriminator(ngpu).to(device)
    netD.load_state_dict(torch.load(f"final_results/cGAN/disc{size_z}_{step-1}.pt"))

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
if step == 0:
    netD.apply(weights_init)

# Print the model
print(netD)


#### Initialisation

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(batch_size, nz, 64, 64, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))


## Training

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[1].to(device)
        truth = data[0].to(device)
        if epoch == 0 and i == 0:
            fixed_truth = truth
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype = torch.float, device = device)
        # Forward pass real batch through D
        output = netD(real_cpu, truth).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 64, 64, device = device)
        # Generate fake image batch with G
        #print(noise.size(), truth.size())
        fake = netG(noise, truth)[:, :, :64, :64]
        label.fill_(fake_label)
        # Classify all fake batch with D
        #print(fake.size(), truth.size())
        output = netD(fake.detach(), truth).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, truth).view(-1)
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
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item()) 

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fixed_fake = netG(fixed_noise, fixed_truth).detach().cpu()
            img_list.append(vutils.make_grid(fixed_fake[ :, :, :64, :64], padding = 2, normalize = True))

        iters += 1

torch.save(netD.state_dict(), f"final_results/cGAN/discinv{size_z}_{step}.pt")
torch.save(netG.state_dict(), f"final_results/cGAN/geninv{size_z}_{step}.pt")


## Results

print()
print("Training time : {}".format(time.time() - t0))
print()

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"final_results/cGAN/lossinv{size_z}_{step}.png")

"""
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated = True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval = 1000, repeat_delay = 1000, blit = True)

HTML(ani.to_jshtml())
"""

# Grab a batch of real images from the dataloader

real_batch = next(iter(dataloader))

"""
# Plot the real images
plt.figure(figsize = (15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imsave("results/Creal_image4.png", np.transpose(vutils.make_grid(fixed_truth, padding = 2, normalize = True).cpu(),(1,2,0)).numpy())
"""

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imsave(f"final_results/cGAN/fake_imageinv{size_z}_{step}.png", np.transpose(img_list[-1],(1,2,0)).numpy())