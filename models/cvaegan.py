import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import BodyMapDataset
from torch.utils.data import DataLoader
from torch.nn import init
from torch.optim import lr_scheduler
from cv2 import imread, imwrite, connectedComponents
from torchsummary import summary
from models.networks import *
from models import utils
from visdom import Visdom
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

        
class CVAE_T(torch.nn.Module):
    def __init__(self, args, encoder, decoder):
        super(CVAE_T, self).__init__()
        self.z_dim = args.z_dim
        self.encoder = encoder
        self.decoder = decoder
        self.device = args.device

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        # print(h_enc.shape)
        mu = h_enc[:, :self.z_dim]
        log_sigma = h_enc[:, self.z_dim:]
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).type(torch.FloatTensor).to(self.device)

        self.z_mean = mu
        self.z_sigma = sigma

        # print(self.device, type(mu), type(sigma), type(std_z))
        # print(mu.shape, sigma.shape, std_z.shape)

        return mu + sigma * std_z

    def forward(self, state, label1, label2):
        h_enc = self.encoder(state, label1)
        z = self._sample_latent(h_enc)
        return self.decoder(z, label2)

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class CVAE(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 10
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.model_name = args.gan_type
        self.device = args.device
        self.y_dim = args.y_dim
        self.z_dim = args.z_dim
        self.pix_dim = args.pix_dim

        # data process
        # Load datasets
        self.train_data = BodyMapDataset(data_root=args.data_dir, dataset=args.dataset, phase='train', max_size=args.data_size, dim=args.pix_dim, cls_num=args.y_dim)
        self.test_data = BodyMapDataset(data_root=args.data_dir, dataset=args.dataset, phase='test', max_size=args.data_size, dim=args.pix_dim, cls_num=args.y_dim)

        print("Trainset: %d | Testset: %d \n"%(len(self.train_data), len(self.test_data)))

        self.trainset_loader = DataLoaderX(
            dataset=self.train_data,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=args.worker,
            pin_memory=True
        )
        # self.testset_loader = DataLoaderX(
        #     dataset=self.test_data,
        #     batch_size=args.batch_size,
        #     drop_last=True,
        #     shuffle=True,
        #     num_workers=args.worker,
        #     pin_memory=True
        # )

        # networks init
        self.En = encoder(args, self.dataset)
        self.De = decoder(args, self.dataset)
        self.CVAE = CVAE_T(args, self.En, self.De)
        self.CVAE_optimizer = optim.Adam(self.CVAE.parameters(),lr=args.lrG, betas=(0.5, 0.999))

        # to device
        self.CVAE.to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)

        # load dataset
        self.z_n = utils.gaussian(1, self.z_dim)

        # fixed noise & condition
        # self.sample_z_.shape (100, 62) noise
        self.sample_z_ = torch.zeros((self.sample_num * self.y_dim, self.z_dim))
        for i in range(self.sample_num):
            self.sample_z_[i * self.y_dim] = torch.from_numpy(self.z_n)
            for j in range(1, self.y_dim):
                self.sample_z_[i * self.y_dim + j] = self.sample_z_[i * self.y_dim]

        # self.sample_y_.shape (100, 1)
        temp = torch.linspace(0, self.y_dim-1, self.y_dim).reshape(-1,1)

        temp_y = torch.zeros((self.sample_num * self.y_dim, 1))
        for i in range(self.sample_num):
            temp_y[i * self.y_dim: (i + 1) * self.y_dim] = temp

        # target = torch.randint(high=self.y_dim-1, size=(1, self.batch_size))
        # y = torch.zeros(self.sample_num* self.y_dim, self.y_dim)
        # y[range(y.shape[0]), target]=1
        
        self.sample_y_ = torch.zeros((self.sample_num* self.y_dim, self.y_dim))
        self.sample_y_.scatter_(1, temp_y.type(torch.LongTensor), 1)
            # self.test_labels = y
            
        self.fill = torch.zeros((self.y_dim, self.y_dim, self.pix_dim, self.pix_dim))
        for i in range(self.y_dim):
            self.fill[i, i, :, :] = 1

    def train(self):
        self.train_hist = {}
        self.train_hist['VAE_loss'] = []
        self.train_hist['KL_loss'] = []
        self.train_hist['LL_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.CVAE.train()
        print('training start!!')
        
        # visdom
        viz = Visdom(port=2000, env='VAE')
        assert viz.check_connection()
        iter_plot_loc = utils.create_loc_plot(viz, 'Iteration', 'Loss', 'VAE:(iter)', ['VAE', 'KL', 'LL'])
        epoch_plot_loc = utils.create_loc_plot(viz, 'Epoch', 'Loss', 'VAE:(epoch)', ['VAE', 'KL', 'LL'])
    
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            VAE_loss_total = []
            KL_loss_total = []
            LL_loss_total = []
            for iter, (imgs, labels) in enumerate(self.trainset_loader):
                
                self.En.train()
                self.De.train()
                
                x_ = imgs.to(self.device)
                y_vec_ = labels.to(self.device)
                y_fill_ = self.fill[torch.max(y_vec_, 1)[1].squeeze()].to(self.device)
                
                # print(x_.shape, y_vec_.shape, y_fill_.shape)
             
                # update VAE network
                dec = self.CVAE(x_, y_fill_, y_vec_)
                self.CVAE_optimizer.zero_grad()
                KL_loss = latent_loss(self.CVAE.z_mean, self.CVAE.z_sigma)
                LL_loss = self.L1_loss(dec, x_)
                VAE_loss = LL_loss + KL_loss
                
                VAE_loss_total.append(VAE_loss.item())
                KL_loss_total.append(KL_loss.item())
                LL_loss_total.append(LL_loss.item())

                self.train_hist['VAE_loss'].append(VAE_loss.item())
                self.train_hist['KL_loss'].append(KL_loss.item())
                self.train_hist['LL_loss'].append(LL_loss.item())

                VAE_loss.backward()
                self.CVAE_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f VAE_loss: %.8f KL_loss: %.8f LL_loss: %.8f" %
                          ((epoch + 1), (iter + 1), len(self.train_data) // self.batch_size, time.time() - start_time,
                           VAE_loss.item(), KL_loss.item(), LL_loss.item()))
                    utils.update_loc_plot(viz, iter_plot_loc, "iter", epoch, iter, self.batch_size, [VAE_loss.item(), KL_loss.item(), LL_loss.item()])
                    
                if np.mod((iter + 1), 200) == 0:
                    # print("saving results images...")
                    self.En.eval()
                    self.De.eval()
                    with torch.no_grad():
                        samples = self.De(self.sample_z_, self.sample_y_, 'test')
                    samples = samples.numpy().transpose(0, 2, 3, 1)
                    tot_num_samples = 100
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    utils.save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                utils.check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                                '_train_{:02d}_{:04d}.png'.format(epoch, (iter + 1)))
            if epoch % 10 == 0:
                self.save()
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            utils.update_loc_plot(viz, epoch_plot_loc, "epoch", epoch, iter, self.batch_size, [VAE_loss_total, KL_loss_total, LL_loss_total])

            # self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/'+ self.model_dir + '/'+ self.model_name, self.epoch)
        utils.generate_train_animation(self.result_dir + '/'+ self.model_dir + '/'+ self.model_name, self.epoch)
        utils.loss_VAE_plot(self.train_hist, os.path.join(self.save_dir, self.model_dir), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.De.eval()

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.De(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            temp = torch.LongTensor(self.batch_size, 1).random_() % 10
            sample_y_ = torch.FloatTensor(self.batch_size, self.y_dim)
            sample_y_.zero_()
            sample_y_.scatter_(1, temp, 1)

            with torch.no_grad():
                sample_z_ = torch.from_numpy(self.z_n).to(self.device)
                sample_y_.to(self.device)

            samples = self.De(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.detach().cpu().numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          utils.check_folder(self.result_dir + '/' + self.model_dir) + '/' +
                          self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_data_{}_batch_{}_embed_{}".format(
            self.model_name, self.dataset,
            self.batch_size, self.z_dim)

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.CVAE.state_dict(), os.path.join(save_dir, self.model_name + '_VAE.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.CVAE.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_VAE.pkl')))

