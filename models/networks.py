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
from models.utils import *


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

        return mu + sigma * std_z

    def forward(self, state, label1, label2):
        h_enc = self.encoder(state, label1)
        z = self._sample_latent(h_enc)
        return self.decoder(z, label2)

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return torch.mean(mean_sq + stddev_sq - torch.log(1e-8 + stddev_sq) - 1)

class decoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, args, dataset='1101'):
        super(decoder, self).__init__()
        self.z_dim = args.z_dim
        self.norm_layer = get_norm_layer(layer_type='batch')
        self.nl_layer = get_non_linearity(layer_type='lrelu')
        self.pix_dim = args.pix_dim
        self.in_dim = args.z_dim + args.y_dim

        self.decoder = ConvResDecoder(self.pix_dim, self.in_dim, 64, 6, self.norm_layer, self.nl_layer)
        init_net(self.decoder, 'kaiming')
      
    def forward(self, input, label):
        x = torch.cat([input, label], dim=1)
        x = self.decoder(x)

        return x

class encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, args, dataset = '1101'):
        super(encoder, self).__init__()

        self.z_dim = args.z_dim * 2
        self.norm_layer = get_norm_layer(layer_type='batch')
        self.nl_layer = get_non_linearity(layer_type='lrelu')
        self.pix_dim = args.pix_dim
        self.in_dim = 3 + args.y_dim

        self.encoder = ResNetEncoder(self.pix_dim, self.in_dim, self.z_dim, 64, 6, self.norm_layer, self.nl_layer)
        init_net(self.encoder, 'kaiming')

    def forward(self, input, label):
        x = torch.cat([input, label], dim=1)
        x = self.encoder(x)

        return x


#####  Decoder #####
class ConvResDecoder(nn.Module):
    '''
        ConvResDecoder: Use convres block for upsampling
    '''
    def __init__(self, im_size, nz, ngf=64, nup=6,
        norm_layer=None, nl_layer=None):
        super(ConvResDecoder, self).__init__()
        self.im_size = im_size // (2 ** nup)
        fc_dim = 2 * nz
        
        layers = []
        prev = 8
        for i in range(nup-1, -1, -1):
            cur = min(prev, 2**i)
            layers.append(ConvResBlock(ngf * prev, ngf * cur, direction='up', stride=2,
                norm_layer=norm_layer, activation_layer=nl_layer))
            prev = cur
        
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        ]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(nz, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(fc_dim, self.im_size * self.im_size * ngf * 8),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], -1, self.im_size, self.im_size)
        return self.conv(x)

class ConvUpSampleDecoder(nn.Module):
    '''
        SimpleDecoder
    '''
    def __init__(self, im_size, nz, ngf=64, nup=6,
        norm_layer=None, nl_layer=None):
        super(ConvUpSampleDecoder, self).__init__()
        self.im_size = im_size // (2 ** nup)
        fc_dim = 4 * nz
        
        layers = []
        prev = 8
        for i in range(nup-1, -1, -1):
            cur = min(prev, 2**i)
            layers.append(deconv3x3(ngf * prev, ngf * cur, stride=2))
            prev = cur
        
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        ]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(nz, fc_dim),
            nl_layer,
            nn.Dropout(),
            nn.Linear(fc_dim, self.im_size * self.im_size * ngf * 8),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], -1, self.im_size, self.im_size)
        return self.conv(x)

class ResNetEncoder(nn.Module):
    def __init__(self, im_size, in_dim, nz=256, ngf=64, ndown=6,
        norm_layer=None, nl_layer=None):
        super(ResNetEncoder, self).__init__()
        self.ngf = ngf
        fc_dim = 2 * nz

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_dim, ngf, kernel_size=7, stride=1, padding=0),
            norm_layer(ngf),
            nl_layer,
        ]
        prev = 1
        for i in range(ndown):
            im_size //= 2
            cur = min(8, prev*2)
            layers.append(ConvResBlock(ngf * prev, ngf * cur, direction='down', stride=2,
                norm_layer=norm_layer, activation_layer=nl_layer))
            prev = cur

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(im_size * im_size * ngf * cur, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nl_layer,
            nn.Linear(fc_dim, nz)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

class ConvResBlock(nn.Module):
    def __init__(self, inplanes, planes, direction, stride=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(0.2, True)):
        super(ConvResBlock, self).__init__()
        self.res = BasicResBlock(inplanes, norm_layer=norm_layer,
                                 activation_layer=activation_layer)
        self.activation = activation_layer

        if stride == 1 and inplanes == planes:
            conv = lambda x: x
        else:
            if direction == 'down':
                conv = conv3x3(inplanes, planes, stride=stride)
            elif direction == 'up':
                conv = deconv3x3(inplanes, planes, stride=stride)
            else:
                raise (ValueError('Direction must be either "down" or "up", get %s instead.' % direction))
        self.conv = conv
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

    def forward(self, x):
        return self.conv(self.activation(self.res(x)))


class BasicResBlock(nn.Module):
    def __init__(self, inplanes, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.LeakyReLU(0.2, True)):
        super(BasicResBlock, self).__init__()

        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.inplanes = inplanes

        layers = [
            conv3x3(inplanes, inplanes),
            norm_layer(inplanes),
            activation_layer,
            conv3x3(inplanes, inplanes),
            norm_layer(inplanes)
        ]
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return self.res(x) + x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

def deconv3x3(in_planes, out_planes, stride=1):

    return nn.Sequential(
        Interpolate(scale_factor=stride, mode='bilinear'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes,
                  kernel_size=3, stride=1, padding=0)
    )

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
def init_net(net, init_type='normal'):
    net = nn.DataParallel(net)
    init_weights(net, init_type)
    return net.to('cuda')

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif layer_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer
    
def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = nn.ReLU(inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = nn.LeakyReLU(0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = nn.ELU(inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer