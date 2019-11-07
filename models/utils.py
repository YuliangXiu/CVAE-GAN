import os, gzip, torch
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import skimage

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, pix_dim, image_path):
    image = (np.squeeze(merge(images, size))+1.0)*127.5
    return cv2.imwrite(image_path, image)

def save_images_test(in_images, out_images, iter_num, batch_size, image_size, image_path):
    vis_image = np.zeros((image_size[0]*2*iter_num, batch_size*image_size[1], 3))
    for iter in range(iter_num):
        for idx in range(batch_size):
            vis_image[(iter*2+0)*image_size[0]:(iter*2+1)*image_size[0], idx*image_size[1]:(idx+1)*image_size[1]] = in_images[iter][idx]
            vis_image[(iter*2+1)*image_size[0]:(iter*2+2)*image_size[0], idx*image_size[1]:(idx+1)*image_size[1]] = out_images[iter][idx]
        vis_image[(iter*2+2)*image_size[0]-1] *= 0
    return cv2.imwrite(image_path, (vis_image+1.0)*127.5)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '_test_all_classes.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def generate_train_animation(path, num, id):
    images = []
    for e in range(num):
        img_name = path + '_train_%02d' % (e) + '_%04d.png'%(id)
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_train_animation.gif', images, fps=5)

def loss_plot(hist, path='Train_hist.png', model_name=''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def loss_VAE_plot(hist, path='Train_hist.png', model_name=''):
    x = range(len(hist['VAE_loss']))

    y1 = hist['VAE_loss']
    y2 = hist['KL_loss']
    y3 = hist['LL_loss']

    plt.plot(x, y1, label='VAE_loss')
    plt.plot(x, y2, label='KL_loss')
    plt.plot(x, y3, label='LL_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def gaussian(batch_size, n_dim, mean=0, var=1, n_labels=10, use_label_info=False):
    if use_label_info:
        if n_dim != 2:
            raise Exception("n_dim must be 2.")

        def sample(n_labels):
            x, y = np.random.normal(mean, var, (2,))
            angle = np.angle((x-mean) + 1j*(y-mean), deg=True)

            label = ((int)(n_labels*angle))//360

            if label<0:
                label+=n_labels

            return np.array([x, y]).reshape((2,)), label

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        z_id = np.empty((batch_size, 1), dtype=np.int32)
        for batch in range(batch_size):
            for zi in range((int)(n_dim/2)):
                    a_sample, a_label = sample(n_labels)
                    z[batch, zi*2:zi*2+2] = a_sample
                    z_id[batch] = a_label
        return z, z_id
    else:
        z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
        return z

def multi_gaussian(batch_size, n_dim, mean_var):
    z = np.zeros((batch_size, n_dim))
    for i in range(n_dim):
        z[:,i] = np.random.normal(mean_var['mean'][i], mean_var['std'][i], (batch_size, 1)).astype(np.float32)
    return z

def create_loc_plot(viz, _xlabel, _ylabel, _title, legend):
    return viz.line(
        X=np.zeros((1,len(legend))),
        Y=np.zeros((1,len(legend))),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=legend
        )
    )

def create_vis_plot(viz, _title, batch, dim):
    return viz.images(
        np.random.randn(16, 3, dim, dim),
        opts=dict(
            title=_title, 
        )
    )

def update_vis_plot(viz, window, batch, dec, x):
    
    dec_img = dec.detach().cpu().numpy()[:8]
    x_img = x.detach().cpu().numpy()[:8]
    viz.images(np.concatenate((x_img, dec_img),axis=0), nrow=8, padding=4, win=window)

def update_loc_plot(viz, window, epoch_or_iter, epoch, i, batch_per_epoch, losses):

    if epoch_or_iter == 'epoch':
        x_arr = np.ones((1, ))*(epoch)
        y_arr = torch.cat([torch.mean(torch.Tensor(loss)).unsqueeze(0)for loss in losses]).unsqueeze(0).detach().cpu().numpy()
    elif epoch_or_iter == 'iter':
        x_arr = np.ones((1,len(losses)))*(epoch*batch_per_epoch+i)
        y_arr = torch.Tensor([loss for loss in losses]).unsqueeze(0).detach().cpu().numpy()
    
    viz.line(
        X=x_arr,
        Y=y_arr,
        win=window,
        update='append'
    )    
    
def gaus2d(x=0, y=0, mx=0, my=0, sx=10, sy=10):
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def generate_mask(stretch, dim, cls_num):
    
    matfile = "template_info/template_female_202" + "_stretch"*stretch + ".mat"
    labelfile = "template_info/template_female_202" + "_stretch"*stretch + "_feature_pose_label.mat"
    npyfile = "template_info/mask" + "_stretch"*stretch + ".npy"
    pngfile = "template_info/dist_vis/dist" + "_stretch"*stretch + "_%s.png"
    feature_points = sio.loadmat(matfile, squeeze_me=True)['para']['feature_uv'].item().astype(np.int32)
    feature_points_label = sio.loadmat(labelfile, squeeze_me=True)['body_pose_index_feature_uv']

    if os.path.exists(npyfile):
        mask = np.load(npyfile)
    else:
  
        mask = np.ones((cls_num, dim, dim))
        for cls_id in range(cls_num):
            for keypoint in feature_points[(feature_points_label[cls_id+1]-1).tolist(),:]:
                mask[cls_id, int((keypoint[1]-1)/(512/dim)), int((keypoint[0]-1)/(512/dim))] = 0

            dist = cv2.distanceTransform(mask[cls_id].astype(np.uint8), cv2.DIST_L2, 3)
            cv2.normalize(dist, dist, 0, 1, cv2.NORM_MINMAX)
            cv2.imwrite(pngfile%(cls_id+2), 255*(1-dist))
            mask[cls_id, :, :] = 1-dist

        np.save(npyfile, mask)
    
    return torch.Tensor(mask)

def generate_mask_guass(stretch, dim, cls_num):
    
    matfile = "template_info/template_female_202" + "_stretch"*stretch + ".mat"
    labelfile = "template_info/template_female_202" + "_stretch"*stretch + "_feature_pose_label.mat"
    npyfile = "template_info/mask_guass" + "_stretch"*stretch + ".npy"
    pngfile = "template_info/guass_vis/guass" + "_stretch"*stretch + "_%s.png"
    feature_points = sio.loadmat(matfile, squeeze_me=True)['para']['feature_uv'].item().astype(np.int32)
    feature_points_label = sio.loadmat(labelfile, squeeze_me=True)['body_pose_index_feature_uv']

    if os.path.exists(npyfile):
        mask = np.load(npyfile)
    else:
  
        mask = np.zeros((cls_num, dim, dim))
        x = np.linspace(0, dim-1, dim)
        y = np.linspace(0, dim-1, dim)
        x, y = np.meshgrid(x, y)

        std = dim/10.0

        for cls_id in range(cls_num):
            for keypoint in feature_points[(feature_points_label[cls_id+1]-1).tolist(),:]:
                mask[cls_id] += gaus2d(x, y, int((keypoint[0]-1)/(512/dim)), int((keypoint[1]-1)/(512/dim)), std, std)
            mask[cls_id] = (mask[cls_id]-np.min(mask[cls_id]))/(np.max(mask[cls_id])-np.min(mask[cls_id]))
            cv2.imwrite(pngfile%(cls_id+2), 255*(mask[cls_id]))
        np.save(npyfile, mask)
    
    return torch.Tensor(mask)

if __name__ == '__main__':
    generate_mask(True, 256, 16)
    generate_mask_guass(True, 256, 16)
    