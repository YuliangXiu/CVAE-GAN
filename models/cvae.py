import torch, time, random, os, pickle
import numpy as np
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import BodyMapDataset
from torch.utils.data import DataLoader
from torch.nn import init
from torch.autograd import grad 
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

class CVAE(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 5
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.device = args.device
        self.z_dim = args.z_dim
        self.pix_dim = args.pix_dim

        self.mask = utils.generate_mask_guass_details("stretch" in self.dataset, self.pix_dim).to(self.device)
        
        # Load datasets
        self.train_data = BodyMapDataset(data_root=args.data_dir, max_size=args.data_size, dim=args.pix_dim)

        self.sample_per_batch = len(self.train_data)/self.batch_size
        print("Trainset: %d \n"%(len(self.train_data)))

        self.trainset_loader = DataLoaderX(
            dataset=self.train_data,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=args.worker,
            pin_memory=True
        )
        
        # networks init
        self.En = nn.DataParallel(encoder(args))
        self.De = nn.DataParallel(decoder(args))
        self.PG = nn.DataParallel(NLayerDiscriminator(input_nc=3)).to(self.device)
        self.CVAE = CVAE_T(args, self.En, self.De)
        
        if args.resume:
            self.load()

        if args.testmode:
            self.load(args.pkl)         
            
        self.CVAE_optimizer = optim.Adam([{'params': self.En.parameters(), 'lr' : 1e-4},
                                         {'params': self.De.parameters(), 'lr': 1e-4},
                                         {'params': self.PG.parameters(), 'lr': 1e-3}], 
                                         lr=args.lrG, betas=(0.5, 0.999), eps=1e-08, weight_decay=1e-6)
        # self.CVAE_optimizer = optim.SGD(self.CVAE.parameters(), lr=args.lrG, momentum=0.9, weight_decay=1e-4)
        # self.CVAE_scheduler = torch.optim.lr_scheduler.StepLR(self.CVAE_optimizer, step_size=50, gamma=0.3)
        self.CVAE_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.CVAE_optimizer, mode='min', factor=0.3, patience=5, verbose=True, threshold=1e-4)
        
        # to device
        self.CVAE.to(self.device)

        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)

        self.sample_z_ = None
        self.do_sample(self.sample_num, self.z_dim)

    def do_sample(self, sample_num, z_dim):

        self.sample_z_ = torch.zeros((sample_num*sample_num, z_dim))
        for i in range(sample_num*sample_num):
            self.sample_z_[i] = torch.from_numpy(utils.gaussian(1, z_dim, mean=0.0, var=1.0))
          

    def train(self):
        self.train_hist = {}
        self.train_hist['LL_loss'] = []
        self.train_hist['CLS_loss'] = []
        self.train_hist['PG_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.CVAE.train()
        print('training start!!')
        
        # visdom
        viz = Visdom(port=2000)
        if self.dataset in viz.get_env_list():
            viz.delete_env(self.dataset)
        viz = Visdom(port=2000, env=self.dataset)
        assert viz.check_connection()


        iter_plot_loc = utils.create_loc_plot(viz, 'Iteration', 'Loss', '%s:(iter)'%(self.dataset), ['LL', 'CLS', 'PG'])
        epoch_plot_loc = utils.create_loc_plot(viz, 'Epoch', 'Loss', '%s:(epoch)'%(self.dataset), ['LL', 'CLS', 'PG'])
        iter_plot_vis = utils.create_vis_plot(viz, 'Iter-Visualization', self.batch_size, self.pix_dim)
    
        start_time = time.time()

        for epoch in range(self.epoch):
            
            epoch_start_time = time.time()
            
            LL_loss_total = []
            CLS_loss_total = []
            PG_loss_total = []
            
            self.En.train()
            self.De.train()
            
            for iter, (imgs, labels) in enumerate(self.trainset_loader):
                
                x_ = imgs.to(self.device)
                labels = labels.to(self.device)
                             
                # update VAE network
                enc, dec = self.CVAE(x_, labels)
                PG_fake = self.PG(dec)
                PG_real = self.PG(x_)

                # summary(self.En.module, (3,256,256))
                # summary(self.De.module, (510,))
                
                LL_loss = self.L1_loss(dec*self.mask, x_*self.mask)/18.0
                CLS_loss = self.MSE_loss(enc, labels)
                PG_real_loss = self.L1_loss(PG_real, torch.Tensor([1.0]).expand_as(PG_real).to(self.device))
                PG_fake_loss = self.L1_loss(PG_fake, torch.Tensor([0.0]).expand_as(PG_fake).to(self.device))
                PG_loss = (PG_real_loss +  PG_fake_loss)*0.5
                                
                LL_loss_total.append(LL_loss.item())
                CLS_loss_total.append(CLS_loss.item())
                PG_loss_total.append(PG_loss.item())

                self.train_hist['LL_loss'].append(LL_loss.item())
                self.train_hist['CLS_loss'].append(CLS_loss.item())
                self.train_hist['PG_loss'].append(PG_loss.item())
                
                # LL_loss.backward(retain_graph=True) 
                # CLS_loss.backward(retain_graph=True)
                # PG_loss.backward()

                LL_loss.backward(retain_graph=True) 
                CLS_loss.backward()

                self.CVAE_optimizer.step()
                self.CVAE_optimizer.zero_grad()

                if ((iter + 1) % (10)) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f LL_loss: %.8f CLS_loss: %.8f PG_loss: %.8f" %
                          ((epoch + 1), (iter + 1), len(self.train_data) // self.batch_size, time.time() - start_time,
                           LL_loss.item(), CLS_loss.item(), PG_loss.item()))
                    utils.update_loc_plot(viz, iter_plot_loc, "iter", epoch, iter, self.sample_per_batch, 
                                          [LL_loss.item(), CLS_loss.item(), PG_loss.item()])
                    utils.update_vis_plot(viz, iter_plot_vis, self.batch_size, dec, x_)
                    # self.save()
            
            if epoch % 1 == 0:
                self.save()
                # torch.cuda.empty_cache()
               

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            utils.update_loc_plot(viz, epoch_plot_loc, "epoch", epoch, iter, self.sample_per_batch, 
                                  [LL_loss_total, CLS_loss_total, PG_loss_total])

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        # utils.generate_train_animation(self.result_dir + '/'+ self.model_dir + '/'+ self.dataset, self.epoch, len(self.train_data)/self.batch_size)
        utils.loss_VAE_plot(self.train_hist, os.path.join(self.save_dir, self.model_dir), self.dataset)

        # change the lr
        self.CVAE_scheduler.step(torch.mean(torch.FloatTensor(LL_loss_total)))
    
    @staticmethod
    def gpu2cpu(tensor):

        mean_ = np.array([-0.03210393, -0.02885828,  0.02909984]) 
        std_ =np.array([0.10642165, 0.08386147, 0.11332943])
        min_ = np.array([-10.62554399,  -9.843649  , -10.25687804]) 
        max_ = np.array([ 6.51756452,  9.55840837, 10.42095193])

        return (tensor.detach().cpu().numpy().transpose(1,2,0)*(max_-min_)+min_)*std_+mean_
    
    @staticmethod
    def gpu2img(tensor):
        return tensor.detach().cpu().numpy().transpose(0,2,3,1)*255.0


    def test(self, flag='ED'):
        import itertools
        
        if flag == 'ED':

            # test Encoder-Decoder
            self.En.eval()
            self.De.eval()
            
            with torch.no_grad():

                middle_num = 10
                outs = []
                ins = []
                for iter, (imgs, labels) in enumerate(self.trainset_loader):

                    x_ = imgs.to(self.device)
                    latent_vec = self.En(x_)

                    pair = list(itertools.combinations(range(self.batch_size),2))
                    for (start,end) in pair:
                        comb = np.zeros((4+middle_num, 256, 256, 3))
                        comb_latent = torch.zeros(2+middle_num, 51)
                        
                        start_vec = latent_vec[start][None, ...]
                        end_vec = latent_vec[end][None, ...]

                        # latent vecotr save
                        comb_latent[0] = start_vec
                        comb_latent[-1] = end_vec

                        start_img = self.De(start_vec)
                        end_img = self.De(end_vec)

                        comb[0], comb[1], comb[-2], comb[-1] = self.gpu2cpu(x_[start]), self.gpu2cpu(start_img[0]), self.gpu2cpu(end_img[0]), self.gpu2cpu(x_[end])

                        for mid in range(middle_num):
                            mid_vec = end_vec * ((mid+1)/middle_num) + start_vec * ((middle_num-mid-1)/middle_num)
                            middle_img = self.De(mid_vec)
                            comb[2+mid] = self.gpu2cpu(middle_img[0])

                            comb_latent[1+mid] = mid_vec

                        utils.check_folder(self.result_dir + '/' + self.model_dir + '/middle_samples_long/')
                        utils.check_folder(self.result_dir + '/' + self.model_dir + '/middle_samples_short/')

                        cv2.imwrite(self.result_dir + '/' + self.model_dir + '/middle_samples_long/' +
                        '_iter_{:03d}_start_{:03d}_end_{:03d}.png'.format(iter, start, end), ((comb+0.3)*127.5).transpose(1,0,2,3).reshape(256, 256*(middle_num+4), 3))

                        utils.save_mats(self.result_dir + '/' + self.model_dir + '/middle_samples_short/' +
                        '_iter_{:03d}_start_{:03d}_end_{:03d}_mid_{:03d}.mat', iter, start, end, comb)
                        
                        # np.save(self.result_dir + '/' + self.model_dir + '/middle_samples_long/' +
                        # '_iter_{:03d}_start_{:03d}_end_{:03d}.npy'.format(iter, start, end), comb_latent.detach().cpu().numpy())
                        # break 
                    break


    @property
    def model_dir(self):
        return "{}_pix_{}_batch_{}_embed_{}".format(
            self.dataset, self.pix_dim, self.batch_size, self.z_dim)

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.En.module.state_dict(), os.path.join(save_dir, self.dataset + '_encoder.pkl'))
        torch.save(self.De.module.state_dict(), os.path.join(save_dir, self.dataset + '_decoder.pkl'))
        torch.save(self.PG.module.state_dict(), os.path.join(save_dir, self.dataset + '_patchD.pkl'))

        with open(os.path.join(save_dir, self.dataset + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self, pkl=None):
        if pkl is None:
            save_dir = os.path.join(self.save_dir, self.model_dir)
            self.En.module.load_state_dict(torch.load(os.path.join(save_dir, self.dataset + '_encoder.pkl')))
            self.De.module.load_state_dict(torch.load(os.path.join(save_dir, self.dataset + '_decoder.pkl')))
            self.PG.module.load_state_dict(torch.load(os.path.join(save_dir, self.dataset + '_patchD.pkl')))
        else:
            self.En.module.load_state_dict(torch.load(pkl + '_encoder.pkl'))
            self.De.module.load_state_dict(torch.load(pkl + '_decoder.pkl'))
            self.PG.module.load_state_dict(torch.load(pkl + '_patchD.pkl'))

