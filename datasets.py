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
    transforms.Normalize([-0.05437761, -0.04876839,  0.05751688], [0.10402182, 0.09962941, 0.11785846]),
    transforms.Normalize([-7.39823482, -7.42358403, -6.69157885], np.array([6.70985841, 6.54264821, 8.95325923])-np.array([-7.39823482, -7.42358403, -6.69157885]))
])


class BodyMapDataset(Dataset):
    def __init__(self, data_root, dataset, dim=256, max_size=-1, device=None, transform=im_trans, cls_num=12):
        
        super(BodyMapDataset, self).__init__()
        # Set image transforms and device
        self.device = torch.device('cuda') if device is None else device
        self.transform = transform
        self.pix_dim = (dim, dim, 3)
        self.cls_num = cls_num
        self.im_root = data_root
        
        # Prepare train/test split
        self.names_unit = np.loadtxt(os.path.join(self.im_root, 'input_PoseUnit.txt'), dtype=str)
        self.names_unit = np.array(["PoseUnit_stretch/"+name_unit for name_unit in self.names_unit], dtype=str)
        self.names_random = np.loadtxt(os.path.join(self.im_root, 'input_PoseRandom.txt'), dtype=str)
        self.names_random = np.array(["PoseRandom_stretch/"+name_random for name_random in self.names_random], dtype=str)
        self.names = np.concatenate((self.names_unit, self.names_random), axis=0)

        self.len =  len(self.names)
        self.im_names = [os.path.join(self.im_root, 'npys', im_name) for im_name in self.names]
        self.w_names = [os.path.join(self.im_root, 'weights', im_name[:-8]+".mat") for im_name in self.names]
    
    def __getitem__(self, id):

        img = self.transform(torch.Tensor(np.load(self.im_names[id]).transpose(2,0,1)))
        label = torch.FloatTensor(sio.loadmat(self.w_names[id])['theta'].flatten())

        return img, label
        
    def __len__(self):
        return len(self.names)


if __name__ == '__main__':

    train_dataset = BodyMapDataset(data_root="./data", dataset="PoseRandom-stretch", cls_num=51, dim=256)
    trainset_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        drop_last=True,
        shuffle=True,
        num_workers=10
    )
    from tqdm import tqdm
    mat = torch.zeros(4150, 51)
    for idx, (img, label) in enumerate(tqdm(trainset_loader)):
        mat[idx] = label.flatten()
    np.save("mean-var.npy", {'mean':torch.mean(mat,dim=0).numpy(), 'std':torch.std(mat, dim=0).numpy()})

    print(torch.mean(mat, dim=0), torch.std(mat, dim=0))
    
    
    
