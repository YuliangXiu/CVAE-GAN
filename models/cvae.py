import torch, time, random, os, pickle
import numpy as np
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
        self.train_data = BodyMapDataset(data_root=args.data_dir, dataset=args.dataset, max_size=args.data_size, dim=args.pix_dim)

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
        self.En = nn.DataParallel(encoder(args, self.dataset))
        self.De = nn.DataParallel(decoder(args, self.dataset))
        self.CVAE = CVAE_T(args, self.En, self.De)
        
        if args.resume:
            self.load()

        if args.testmode:
            self.load(args.pkl)         
            
        self.CVAE_optimizer = optim.Adam([{'params': self.En.parameters(), 'lr' : 1e-5},
                                         {'params': self.De.parameters(), 'lr': 1e-4}], 
                                         lr=args.lrG, betas=(0.5, 0.999), eps=1e-08, weight_decay=1e-4)
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
        self.train_hist['VAE_loss'] = []
        self.train_hist['KL_loss'] = []
        self.train_hist['LL_loss'] = []
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


        iter_plot_loc = utils.create_loc_plot(viz, 'Iteration', 'Loss', '%s:(iter)'%(self.dataset), ['VAE', 'KL', 'LL', 'En-LL', 'En-KL'])
        epoch_plot_loc = utils.create_loc_plot(viz, 'Epoch', 'Loss', '%s:(epoch)'%(self.dataset), ['VAE', 'KL', 'LL'])
        iter_plot_vis = utils.create_vis_plot(viz, 'Iter-Visualization', self.batch_size, self.pix_dim)
    
        start_time = time.time()

        for epoch in range(self.epoch):
            
            epoch_start_time = time.time()
            VAE_loss_total = []
            KL_loss_total = []
            LL_loss_total = []
            
            self.En.train()
            self.De.train()
            
            for iter, imgs in enumerate(self.trainset_loader):
                
                x_ = imgs.to(self.device)
                             
                # update VAE network
                dec = self.CVAE(x_)
                KL_loss = latent_loss(self.CVAE.z_mean, self.CVAE.z_sigma)/(self.batch_size)*1.0
                LL_loss = self.MSE_loss(dec*self.mask, x_*self.mask)/(self.batch_size)*1e2
                VAE_loss = LL_loss + KL_loss
                
                VAE_loss_total.append(VAE_loss.item())
                KL_loss_total.append(KL_loss.item())
                LL_loss_total.append(LL_loss.item())

                self.train_hist['VAE_loss'].append(VAE_loss.item())
                self.train_hist['KL_loss'].append(KL_loss.item())
                self.train_hist['LL_loss'].append(LL_loss.item())
                
                (LL_loss*1e2).backward(retain_graph=True)
                En_params_LL = torch.Tensor([(param.grad**2).mean() for param in list(self.En.parameters())])
                self.En.zero_grad()
                KL_loss.backward(retain_graph=True)
                En_params_KL = torch.Tensor([(param.grad**2).mean() for param in list(self.En.parameters())])
                self.En.zero_grad()
                

                VAE_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.CVAE.parameters(), 1e-1)
                self.CVAE_optimizer.step()
                self.CVAE_optimizer.zero_grad()

                if ((iter + 1) % (10)) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f VAE_loss: %.8f KL_loss: %.8f LL_loss: %.8f En_Grad_LL: %.8f En_Grad_KL: %.8f" %
                          ((epoch + 1), (iter + 1), len(self.train_data) // self.batch_size, time.time() - start_time,
                           VAE_loss.item(), KL_loss.item(), LL_loss.item(), En_params_LL.mean(), En_params_KL.mean()))
                    utils.update_loc_plot(viz, iter_plot_loc, "iter", epoch, iter, self.sample_per_batch, [VAE_loss.item(), KL_loss.item(), LL_loss.item(), En_params_LL.mean(), En_params_KL.mean()])
                    utils.update_vis_plot(viz, iter_plot_vis, self.batch_size, dec, x_)
            
            if epoch % 3 == 0:
                self.save()
                # test samples after every epoch
                torch.cuda.empty_cache()
                # with torch.no_grad():
                #     self.De.eval()
                #     samples = self.De(self.sample_z_)

                # samples = samples.detach().cpu().numpy().transpose(0, 2, 3, 1)
                # tot_num_samples = self.sample_num * self.sample_num
                # manifold_h = self.sample_num
                # manifold_w = self.sample_num
                # utils.save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w], self.pix_dim,
                #             utils.check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.dataset +
                #             '_train_{:02d}_{:04d}.png'.format(epoch, (iter + 1)))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            utils.update_loc_plot(viz, epoch_plot_loc, "epoch", epoch, iter, self.sample_per_batch, [VAE_loss_total, KL_loss_total, LL_loss_total])

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        # utils.generate_train_animation(self.result_dir + '/'+ self.model_dir + '/'+ self.dataset, self.epoch, len(self.train_data)/self.batch_size)
        utils.loss_VAE_plot(self.train_hist, os.path.join(self.save_dir, self.model_dir), self.dataset)

        # change the lr
        self.CVAE_scheduler.step(torch.mean(torch.FloatTensor(VAE_loss_total)))
    
    # random
    @staticmethod
    def gpu2cpu(tensor):
        mean_ = np.array([-0.05437761, -0.04876839,  0.05751688]) 
        std_ =np.array([0.10402182, 0.09962941, 0.11785846])
        min_ = np.array([-7.39823482, -7.42358403, -6.69157885]) 
        max_ = np.array([6.70985841, 6.54264821, 8.95325923])

        return (tensor.detach().cpu().numpy().transpose(1,2,0)*(max_-min_)+min_)*std_+mean_

    # # poseunit
    # @staticmethod
    # def gpu2cpu(tensor):
    #     mean_ = np.array([-0.00985059, -0.00904417,  0.00027268]) 
    #     std_ =np.array([0.10433362, 0.05872985, 0.0996523 ])
    #     min_ = np.array([-11.05148387, -14.44375436, -11.41507555]) 
    #     max_ = np.array([ 6.43471079, 13.33603401, 12.70854703])

    #     return (tensor.detach().cpu().numpy().transpose(1,2,0)*(max_-min_)+min_)*std_+mean_


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
                for iter, imgs in enumerate(self.trainset_loader):
                    x_ = imgs.to(self.device)
                    latent_vec = self.En(x_)
                    pair = list(itertools.combinations(range(self.batch_size),2))
                    for (start,end) in pair:
                        comb = np.zeros((4+middle_num, 256, 256, 3))
                        comb_latent = torch.zeros(2+middle_num, 52*2)
                        
                        start_vec = latent_vec[start][None, ...]
                        end_vec = latent_vec[end][None, ...]

                        # latent vecotr save
                        comb_latent[0] = start_vec
                        comb_latent[-1] = end_vec

                        start_img = self.De(self.CVAE._sample_latent(start_vec))
                        end_img = self.De(self.CVAE._sample_latent(end_vec))

                        comb[0], comb[1], comb[-2], comb[-1] = self.gpu2cpu(x_[start]), self.gpu2cpu(start_img[0]), self.gpu2cpu(end_img[0]), self.gpu2cpu(x_[end])

                        for mid in range(middle_num):
                            mid_vec = end_vec * ((mid+1)/middle_num) + start_vec * ((middle_num-mid-1)/middle_num)
                            middle_img = self.De(self.CVAE._sample_latent(mid_vec))
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

        with open(os.path.join(save_dir, self.dataset + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self, pkl=None):
        if pkl is None:
            save_dir = os.path.join(self.save_dir, self.model_dir)
            self.En.module.load_state_dict(torch.load(os.path.join(save_dir, self.dataset + '_encoder.pkl')))
            self.De.module.load_state_dict(torch.load(os.path.join(save_dir, self.dataset + '_decoder.pkl')))
        else:
            self.En.module.load_state_dict(torch.load(pkl + '_encoder.pkl'))
            self.De.module.load_state_dict(torch.load(pkl + '_decoder.pkl'))

