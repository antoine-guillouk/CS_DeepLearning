from generator import generator
from discriminator import discriminator
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from utils.img_dataset import ImgDataset
from utils.augmentation import get__augmentation
from utils.load_config import load_config

config = load_config()
dataroot = config["dataroot"]

ground_truth_dir = os.path.join(dataroot, 'train_unmasked')
masked_img_dir = os.path.join(dataroot, 'train')
mask_map_dir = os.path.join(dataroot,"train_labels")

epochs=100
Batch_Size=64
lr=0.0002
beta1=0.5
over=4

try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

dataset = ImgDataset(ground_truth_dir, masked_img_dir, mask_map_dir, augmentation = get__augmentation())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size,
                                         shuffle=True)

ngpu = 1

wtl2 = 0.999

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch=0

netG = generator()
netG.apply(weights_init)


netD = discriminator()
netD.apply(weights_init)

print(netG)
print(netD)


criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(Batch_Size, 3, 128, 128)
input_masked = torch.FloatTensor(Batch_Size, 3, 128, 128)
label = torch.FloatTensor(Batch_Size)
real_label = 1
fake_label = 0


netD.to(device)
netG.to(device)
criterion.to(device)
criterionMSE.to(device)
input_real, input_masked,label = input_real.to(device),input_masked.to(device), label.to(device)


input_real = Variable(input_real)
input_masked = Variable(input_masked)
label = Variable(label)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(resume_epoch,epochs):
    for i, data in enumerate(dataloader, 0):
        
        real_cpu, masked_cpu = data
        batch_size = real_cpu.size(0)
        with torch.no_grad():
            input_real.resize_(real_cpu.size()).copy_(real_cpu)
            input_masked.resize_(masked_cpu.size()).copy_(masked_cpu)

        #start the discriminator by training with real data---
        netD.zero_grad()
        with torch.no_grad():
            label.resize_(batch_size).fill_(real_label)
        output = netD(input_real).view([-1])
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train the discriminator with fake data---
        fake = netG(input_masked)
        label.data.fill_(fake_label)
        output = netD(fake.detach()).view([-1])
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()


        #train the generator now---
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake).view([-1])
        errG_D = criterion(output, label)

        wtl2Matrix = input_real.clone()
        wtl2Matrix.data.fill_(wtl2*10)
        wtl2Matrix.data[:,:,int(over):int(128/2 - over),int(over):int(128/2 - over)] = wtl2

        errG_l2 = (fake-input_real).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()

        errG = (1-wtl2) * errG_D + wtl2 * errG_l2

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d / %d][%d / %d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch, epochs, i, len(dataloader),
                 errD.data, errG_D.data,errG_l2.data, D_x,D_G_z1, ))

        if i % 10 == 0:

            vutils.save_image(real_cpu,
                    'result/train/real/real_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(input_masked.data,
                    'result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            recon_image = input_masked.clone()
            recon_image.data = fake.data
            vutils.save_image(recon_image.data,
                    'result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))