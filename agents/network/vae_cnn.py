import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image

from utilities.util import compute_conv_out_node_num, compute_conv_out_width


class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()
        self.encoder_conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
        )
        self.encoder_linear_layer = nn.Sequential(
            nn.Linear(400, 200)
        )

        self.decoder_linear_layer = nn.Sequential(
            nn.Linear(20, 200),
            nn.ReLU(True),
            nn.Linear(200, 400),
            nn.ReLU(True)
        )
        self.decoder_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )
        self.fc31 = nn.Linear(200, 20)  # 均值
        self.fc32 = nn.Linear(200, 20)  # 方差
        self.batch_size = None

    def encoder(self, x):
        h1_conv = self.encoder_conv_layer(x)
        self.batch_size = h1_conv.size()[0]
        h1_linear = h1_conv.view(self.batch_size, -1)
        h2 = self.encoder_linear_layer(h1_linear)
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar

    def decoder(self, z):
        h3_linear = self.decoder_linear_layer(z)
        h3_conv = h3_linear.view(self.batch_size, 16, 5, 5)
        x = self.decoder_conv_layer(h3_conv)
        return x

    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return mu, logvar, self.decoder(z)


class VAE_CNN_P3D(nn.Module):
    def __init__(self):
        super(VAE_CNN_P3D, self).__init__()
        in_channels = 15
        out_channels = 24
        kernel_size = 4
        stride = 2
        padding = 1
        w = 36
        h = 18
        out_w = compute_conv_out_width(i=w, k=kernel_size, s=stride, p=padding)
        out_h = compute_conv_out_width(i=h, k=kernel_size, s=stride, p=padding)
        out_node_num = compute_conv_out_node_num(d=out_channels, w=out_w, h=out_h)

        self.encoder_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),  # (b, 24, 9, 18)
            nn.ReLU(True),
        )
        self.encoder_linear_layer = nn.Sequential(
            nn.Linear(out_node_num, 512),
            nn.ReLU(True),
        )
        self.fc_mu = torch.nn.Linear(512, 128)
        self.fc_std = torch.nn.Linear(512, 128)

        self.decoder_linear_layer = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, out_node_num),
            nn.ReLU(True)
        )
        # 24 * 9 * 18
        self.decoder_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),  # (b, 8, 16, 16)
            nn.Tanh()
        )

    def encoder(self, x):
        h1_conv = self.encoder_conv_layer(x)
        self.batch_size = h1_conv.size()[0]
        h1_linear = h1_conv.view(self.batch_size, -1)
        h2 = self.encoder_linear_layer(h1_linear)
        mu = self.fc_mu(h2)
        logvar = self.fc_std(h2)
        return mu, logvar

    def decoder(self, z):
        h3_linear = self.decoder_linear_layer(z)
        h3_conv = h3_linear.view(self.batch_size, 24, 8, 8)
        x = self.decoder_conv_layer(h3_conv)
        return x

    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return mu, logvar, self.decoder(z)


class VAE_CNN_CIFAR(nn.Module):
    def __init__(self):
        super(VAE_CNN_CIFAR, self).__init__()
        self.encoder_conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # (b, 16, 5, 5)
        )
        self.encoder_linear_layer = nn.Sequential(
            nn.Linear(1024, 400),
            nn.ReLU(True)
        )

        self.decoder_linear_layer = nn.Sequential(
            nn.Linear(100, 400),
            nn.ReLU(True),
            nn.Linear(400, 1024),
            nn.ReLU(True)
        )
        self.decoder_conv_layer = nn.Sequential(
            # N（out） = （N（in）-1）× s +k -2p
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # (b, 8, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )
        self.fc31 = nn.Linear(400, 100)  # 均值
        self.fc32 = nn.Linear(400, 100)  # 方差
        self.batch_size = None

    def encoder(self, x):
        h1_conv = self.encoder_conv_layer(x)
        self.batch_size = h1_conv.size()[0]
        h1_linear = h1_conv.view(self.batch_size, -1)
        h2 = self.encoder_linear_layer(h1_linear)
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar

    def decoder(self, z):
        h3_linear = self.decoder_linear_layer(z)
        h3_conv = h3_linear.view(self.batch_size, 16, 8, 8)
        x = self.decoder_conv_layer(h3_conv)
        return x

    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准的正态分布中随机采样一个eps
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return mu, logvar, self.decoder(z)
