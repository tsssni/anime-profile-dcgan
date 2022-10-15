import sys
sys.path.append('gan_model')

from gan_model.dcgan import Generator
import torch
import torch.nn as nn
import numpy.random as random
import os
import math

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

feature_size=100
batch_size=64
group_size=100
iters=0

criterion=nn.L1Loss(reduction='mean')

netG=Generator().to(device)
netG.load_state_dict(torch.load(os.path.join('..','dcgan_models','G_PARAM.pth')))

def get_hair_color_fitness(noise,mask):
    mask=(~mask).repeat(batch_size,1)
    orig_pic=netG(noise)
    mask_pic=netG(noise.masked_fill(mask,0))

    orig_hair=torch.zeros(batch_size,3,10,50)
    mask_hair=torch.zeros(batch_size,3,10,50)

    for i in range(batch_size):
        orig_hair[i]=orig_pic[i,:,0:10,7:57]
        mask_hair[i]=mask_pic[i,:,0:10,7:57]

    return criterion(orig_hair,mask_hair).item()   

def get_fitness(noise,group,fitness):
    for i in range(group_size):
        mask=group[i].to(torch.bool)
        fitness[i]=get_hair_color_fitness(noise,mask)
    
    global iters
    print('iter%04d: '%(iters),torch.mean(fitness))
    iters+=1

    if(iters==100):
        torch.save(group,os.path.join('..','mask','mask.pth'))
        sys.exit(0)

def mate(group,prob=0.8):
    mate_range=list()
    for i in range(group_size):
        if(random.rand()<=prob):
            mate_range.append(i)

    if(len(mate_range)%2 == 1):
        mate_range.pop()
    random.shuffle(mate_range)

    for i in range(0,len(mate_range),2):
        mate_bit=math.floor(feature_size*random.rand())
        mate_0=group[mate_range[i],mate_bit:]
        mate_1=group[mate_range[i+1],mate_bit:]
        group[mate_range[i],mate_bit:]=mate_1
        group[mate_range[i+1],mate_bit:]=mate_0

def variation(group,prob=0.01):
    for i in range(group_size):
        if random.rand()<=prob:
            var_bit=math.floor(feature_size*random.rand())
            group[i][var_bit]=not group[i][var_bit]

def roulette_wheel_selection(fitness):
    r=random.random()
    s=0
    idx=-1

    while s<r:
        idx+=1
        s+=fitness[idx]
    return idx

def genetic(group,fitness):
    noise=torch.randn([batch_size,feature_size]).to(device)
    get_fitness(noise,group,fitness)
    normalized_fitness=fitness/torch.sum(fitness)
    new_group=torch.zeros([group_size,feature_size]).to(device)
    
    for i in range(group_size):
        idx=roulette_wheel_selection(normalized_fitness)
        new_group[i]=group[idx]
    
    mate(new_group)
    variation(new_group)
    group=new_group
    genetic(group,fitness)

group=torch.randint(0,2,[group_size,feature_size]).to(device)
fitness=torch.zeros(group_size,dtype=torch.float32).to(device)
genetic(group, fitness)