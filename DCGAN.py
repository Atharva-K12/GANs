import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import loader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
batch_size = 128
image_size = 64
nz = 1000
ng = 64
nd = 64
num_epochs = 10
lr = 0.0002
beta1 = 0.5
dataloader=loader.train_loader_fn(batch_size)
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig('Test_outputs/Sample_data')
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ng * 8, 4, 1, 0),
            nn.BatchNorm2d(ng * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ng * 8, ng * 4, 4, 2, 1),
            nn.BatchNorm2d(ng * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ng * 4, ng * 2, 4, 2, 1),
            nn.BatchNorm2d(ng * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ng * 2, ng, 4, 2, 1),
            nn.BatchNorm2d(ng),
            nn.ReLU(True),
            nn.ConvTranspose2d( ng, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
netG = Generator().to(device)
netG.apply(weights_init)
class Discriminator(nn.Module):
    def __init__(self ):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, nd, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd, nd * 2, 4, 2, 1),
            nn.BatchNorm2d(nd * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd * 2, nd * 4, 4, 2, 1),
            nn.BatchNorm2d(nd * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd * 4, nd * 8, 4, 2, 1),
            nn.BatchNorm2d(nd * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
netD = Discriminator().to(device)
netD.apply(weights_init)
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in tqdm(range(num_epochs)):
    i=0
    for data in tqdm(dataloader):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,),1, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1
    print('epoch :[%d/%d]\tLoss_D: %.8f\tLoss_G: %.8f\n'% (epoch, num_epochs,errD.item(), errG.item()))



plt.figure(figsize=(10,5))
plt.title("Loss")
plt.plot(G_losses,label="Generative")
plt.plot(D_losses,label="Discriminative")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('Test_outputs/Loss')

for j,i in enumerate(img_list):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(i,(1,2,0)))
    plt.savefig('Test_outputs/'+str(j))


real_batch = next(iter(dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig('Test_outputs/Comparison')
plt.show()
torch.save(netD.state_dict(),'discNet.pth')
torch.save(netG.state_dict(),'genNet.pth')