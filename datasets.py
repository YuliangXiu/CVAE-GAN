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
    def __init__(self, data_root, dataset, phase='train', train_test_split=0.98, dim=512, max_size=-1, device=None, transform=im_trans):
        
        super(BodyMapDataset, self).__init__()
        # Set image transforms and device
        self.device = torch.device('cuda') if device is None else device
        self.transform = transform
        self.phase = phase
        self.pix_dim = (dim, dim, 3)
        self.labeldim = 1

        self.im_root = os.path.join(data_root, dataset)
        if not os.path.isdir(self.im_root):
            raise(FileNotFoundError('{} dataset not found, '.format(dataset_name) + 
                'please run "data_washing.py" first'))
        
        # Prepare train/test split
        self.names = np.loadtxt(os.path.join(data_root, '../input_%s.txt'%(dataset)), dtype=str)
        self.len =  len(self.names)
        self.split = train_test_split
        split_point = int(train_test_split * self.len)

        self.im_names = [os.path.join(self.im_root, im_name) for im_name in self.names]

        if self.phase == 'train':
            self.names = self.names[:split_point]
            self.im_names = self.im_names[:split_point]
        elif self.phase == 'test':
            self.names = self.names[split_point:]
            self.im_names = self.im_names[split_point:]
    
    def __getitem__(self, id):

        out_dict = {}
        img = self.transform(resize(imread(self.im_names[id]), self.pix_dim[:-1]))
        label = int(self.names[id].split("_")[1])
        return img, label
        
    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    train_dataset = BodyMapDataset(data_root="./data/GT", phase='train')
    trainset_loader = DataLoader(
        dataset=train_dataset,
        batch_size=10,
        drop_last=True,
        shuffle=True,
        num_workers=10
    )

    for idx, (img, label) in enumerate(trainset_loader):
        print(img.shape)
        print(label.shape)
    
    
    
