import os
import glob

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super(Discriminator,self).__init__()

        def conv_bn_lrelu(in_dim,out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim,out_dim,5,2,2,bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2,inplace=True)
            )

        self.input=nn.Sequential(
            nn.Conv2d(in_channels,dim,5,2,2,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            conv_bn_lrelu(dim,dim*2),
            conv_bn_lrelu(dim*2,dim*4),
            conv_bn_lrelu(dim*4,dim*8),
            nn.Conv2d(dim*8,1,4,1,0),
            nn.Sigmoid()
        )

    def forward(self,input):
        return self.input(input).view(-1) 

class Generator(nn.Module):
    def __init__(self, z_dim=100, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.l1 = nn.Sequential(
            nn.Linear(z_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )

        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self,input):
        output = self.l1(input)
        output = output.view(output.size(0), -1, 4, 4)
        output = self.l2_5(output)
        return output