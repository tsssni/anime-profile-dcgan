import sys
sys.path.append('gan_model')

from gan_model.dcgan import Generator
import os
import math
import random
import torch
import torchvision
import matplotlib.pyplot as plt

feature_size=100
batch_size=64
group_size=100

mask=torch.load(os.path.join('..','mask','mask.pth')).cpu()
random_mask=mask[math.floor(len(mask)*random.random())]

netG=Generator()
netG.load_state_dict(torch.load(os.path.join('..','dcgan_models','G_PARAM.pth')))
netG.eval()
noise=torch.randn(batch_size,feature_size, requires_grad=False)

grid_image=torchvision.utils.make_grid(netG(noise),nrow=8)
plt.figure(figsize=(10,10))
plt.imshow(grid_image.permute(1,2,0))
plt.axis('off')
plt.show()

grid_image=torchvision.utils.make_grid(netG(noise+random_mask*0.5),nrow=8)
plt.figure(figsize=(10,10))
plt.imshow(grid_image.permute(1,2,0))
plt.axis('off')
plt.show()

grid_image=torchvision.utils.make_grid(netG(noise-random_mask*0.25),nrow=8)
plt.figure(figsize=(10,10))
plt.imshow(grid_image.permute(1,2,0))
plt.axis('off')
plt.show()
