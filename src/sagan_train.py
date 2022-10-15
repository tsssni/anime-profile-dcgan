import os
import sys
sys.path.append('gan_model')

import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gan_model.losses import Hinge
from gan_model.sagan import Discriminator,Generator
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

num_epoch=100
begin_epoch=0
batch_size=8
feature_size=100

img_list = []
G_losses = []
D_losses = []

netG=Generator().to(device)
netD=Discriminator().to(device)
netD.load_state_dict(torch.load(os.path.join('..','sagan_models','D_006.pth')))
netG.load_state_dict(torch.load(os.path.join('..','sagan_models','G_006.pth')))
netG.train()
netD.train()

criterion=Hinge().to(device)
optimizerD=torch.optim.RMSprop(netD.parameters(),lr=2e-4)
optimizerG=torch.optim.RMSprop(netG.parameters(),lr=2e-4)

lrd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=5, eta_min=5E-5)
lrg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=5, eta_min=5E-5)

dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
batch_num=len(dataloader)

fixed_noise=torch.randn(batch_size,feature_size)
fixed_noise=fixed_noise.to(device)

for epoch in range(begin_epoch, num_epoch):
    for i,data in enumerate(dataloader,0):
        #----Traing Discriminator---
        optimizerD.zero_grad()
        data=data.to(device)
        
        output_real=netD(data).view(-1)
        noise=torch.randn(len(data),feature_size).to(device)
        with torch.no_grad():
            fake=netG(noise)
        output_fake=netD(fake).view(-1)

        loss_d=criterion(output_real,output_fake)
        loss_d.backward()
        optimizerD.step()
        lrd_scheduler.step()

        #----Training Generator----
        optimizerG.zero_grad()
        noise=torch.randn(len(data),feature_size).to(device)
        fake=netG(noise)

        output=netD(fake).view(-1)
        loss_g=criterion(output)
        loss_g.backward()
        optimizerG.step()

        if i%1000==0:
            print("[%d/%d] [%d/%d] loss_D:%.4f loss_G:%.4f"
             %(epoch,num_epoch,i,batch_num,loss_d.item(),loss_g.item()))

        G_losses.append(loss_g.item())
        D_losses.append(loss_d.item())

    lrg_scheduler.step()
        
    with torch.no_grad():
        fake=netG(fixed_noise)
    grid_image=torchvision.utils.make_grid(fake,nrow=8)
    grid_image=grid_image.cpu()

    plt.figure(figsize=(10,10))
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis('off')

    print('epoch%03d result:\n'%epoch)
    plt.show()

    pathD=os.path.join('..','gan_models','D_{:03d}.pth'.format(epoch))
    pathG=os.path.join('..','gan_models','G_{:03d}.pth'.format(epoch))
    torch.save(netD.state_dict(),pathD)
    torch.save(netG.state_dict(),pathG)