import numpy as np
import torch
from cv2 import imread, resize
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sys import platform
import pickle
import os
from PIL import Image


im_trans = transforms.Compose([
    Image.fromarray,
    transforms.ToTensor(),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
])


class BodyMapDataset(Dataset):
    def __init__(self, data_root, dataset, dim=512, max_size=-1, device=None, transform=im_trans, cls_num=12):
        
        super(BodyMapDataset, self).__init__()
        # Set image transforms and device
        self.device = torch.device('cuda') if device is None else device
        self.transform = transform
        self.pix_dim = (dim, dim, 3)
        self.labeldim = 1
        self.cls_num = cls_num
        self.im_root = os.path.join(data_root, dataset)
        
        # Prepare train/test split
        self.names = np.loadtxt(os.path.join(self.im_root, 'input_%s.txt'%(dataset)), dtype=str)[:max_size]
        self.len =  len(self.names)
        self.im_names = [os.path.join(self.im_root, 'GT_output', im_name) for im_name in self.names]
    
    def __getitem__(self, id):

        img = self.transform(resize(imread(self.im_names[id]), self.pix_dim[:-1]))
        # img = resize(imread(self.im_names[id]), self.pix_dim[:-1]).astype(float)
        label = torch.Tensor([int(self.names[id].split("_")[1])-2]).type(torch.LongTensor)
        y = torch.zeros(1, self.cls_num)
        y[0, label]=1

        return img, y.squeeze(0)
        
    def __len__(self):
        return len(self.names)


if __name__ == '__main__':

    train_dataset = BodyMapDataset(data_root="./data", dataset="PoseUnit", cls_num=16, dim=256)
    trainset_loader = DataLoader(
        dataset=train_dataset,
        batch_size=10,
        drop_last=True,
        shuffle=True,
        num_workers=10
    )

    mean = torch.DoubleTensor([0.,0.,0.]).to('cuda')
    for idx, (img, label) in enumerate(trainset_loader):
        mean = 0.9*mean + 0.1*torch.max(img.to('cuda').permute(0,2,3,1).reshape(-1,3), dim=0)[0] 
        print(mean)
    
    
    
