import glob
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CrypkoDataset(Dataset):
    def __init__(self,fnames,transform):
        self.transform=transform
        self.fnames=fnames
        self.num_samples=len(fnames)

    def __getitem__(self,idx):
        fname=self.fnames[idx]
        image=torchvision.io.read_image(fname)
        image=self.transform(image)
        return image

    def __len__(self):
        return self.num_samples


def get_dataset(path):
    fnames=glob.glob(os.path.join(path,'*'))
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset=CrypkoDataset(fnames,transform)
    return dataset