import os
import sys
sys.path.append('gan_model')

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gan_model.dcgan import Discriminator,Generator
from crypko import get_dataset

device='cuda'  if torch.cuda.is_available() else 'cpu'
print(device)

dataset=get_dataset(os.path.join('..','faces'))
images=[dataset[i] for i in range(64)]
grid_image=torchvision.utils.make_grid(images,nrow=8)
plt.figure(figsize=(10,10))
plt.imshow(grid_image.permute(1,2,0))
plt.axis('off')
plt.show()

num_epoch=200
begin_epoch=0
batch_size=64
feature_size=100

img_list = []
G_losses = []
D_losses = []

netG=Generator().to(device)
netD=Discriminator().to(device)
netG.train()
netD.train()

criterion=nn.BCELoss().to(device)
optimizerG=torch.optim.Adam(netG.parameters(),lr=2e-4,betas=(0.5,0.999))
optimizerD=torch.optim.Adam(netD.parameters(),lr=2e-4,betas=(0.5,0.999))

dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
batch_num=len(dataloader)

real_label=1
fake_label=0

fixed_noise=torch.randn(batch_size,feature_size)
fixed_noise=fixed_noise.to(device)

for epoch in range(begin_epoch, num_epoch):
    for i,data in enumerate(dataloader,0):
        #----Traing Discriminator---
        optimizerD.zero_grad()
        data=data.to(device)
        
        #1:real_label
        labels=torch.full((len(data),),real_label, dtype=torch.float, device=device)
        output=netD(data).view(-1)
        loss_real=criterion(output,labels)
        loss_real.backward()

        #2:fake_label
        noise=torch.randn(len(data),feature_size).to(device)
        with torch.no_grad():
            fake=netG(noise)
            
        labels.fill_(fake_label)
        output=netD(fake).view(-1)
        loss_fake=criterion(output,labels)
        loss_fake.backward()

        loss_d=loss_real+loss_fake
        optimizerD.step()

        #----Training Generator----
        optimizerG.zero_grad()
        noise=torch.randn(len(data),feature_size).to(device)
        fake=netG(noise)

        labels.fill_(real_label)
        output=netD(fake).view(-1)
        loss_g=criterion(output,labels)
        loss_g.backward()
        optimizerG.step()

        if i%1000==0 or i==len(dataloader)-1:
            print("[%d/%d] [%d/%d] loss_D:%.4f loss_G:%.4f"
             %(epoch,num_epoch,i,batch_num,loss_d.item(),loss_g.item()))

        G_losses.append(loss_g.item())
        D_losses.append(loss_d.item())
        
    with torch.no_grad():
        fake=netG(fixed_noise)
    grid_image=torchvision.utils.make_grid(fake,nrow=8)
    grid_image=grid_image.cpu()

    plt.figure(figsize=(10,10))
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis('off')

    print('epoch:%d'%(epoch))
    plt.show()

    pathD=os.path.join('..','gan_models','D_{:03d}.pth'.format(epoch))
    pathG=os.path.join('..','gan_models','G_{:03d}.pth'.format(epoch))
    torch.save(netD.state_dict(),pathD)
    torch.save(netG.state_dict(),pathG)
