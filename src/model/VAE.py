import torch
import torch.nn as nn
from src.models.ResBlock import ResBlockConv, ResBlockConvTranspose
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        # parameters
        self.z_channels = args.z_channels
        self.z_h = args.z_h
        self.z_w = args.z_w
        self.z_dim = self.z_channels * self.z_h * self.z_w
        self.img_dim = args.img_dim
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.channels = args.channels
        self.device = args.device
        
        # posterior
        self.fc1 = nn.Linear(self.z_dim, self.z_dim)
        self.fc2 = nn.Linear(self.z_dim, self.z_dim)
 
        # other
        self.relu = nn.ELU()
        self.kld_weight = 1.0 / 99.0
        
        # reconstruction loss function
        if args.recon_loss == "BCE":
            self.recon_loss = nn.BCELoss(reduction='none')
        elif args.recon_loss == "MSE":
            self.recon_loss = nn.MSELoss(reduction='none')
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden_dim, 4, 2, 1, bias=False), # 8x8 --> 16x16
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 16x16 --> 32x32
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 32x32 --> 64x64
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 64x64 --> 128x128
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 128x128 --> 256x256
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 256x256 --> 512x512
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            ResBlockConv(self.hidden_dim, self.hidden_dim), # 512x512 --> 512x512
            ResBlockConv(self.hidden_dim, self.hidden_dim), # 512x512 --> 512x512
            nn.Conv2d(self.hidden_dim, 1, 3, 1, 1, bias=True) # 512x512 --> 512x512
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, self.hidden_dim, 4, 2, 1, bias=False), # 8x8 --> 16x16
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 16x16 --> 32x32
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 32x32 --> 64x64
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 64x64 --> 128x128
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 128x128 --> 256x256
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False), # 256x256 --> 512x512
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU(),
            ResBlockConvTranspose(self.hidden_dim, self.hidden_dim), # 512x512 --> 512x512
            ResBlockConvTranspose(self.hidden_dim, self.hidden_dim), # 512x512 --> 512x512
            nn.ConvTranspose2d(self.hidden_dim, self.channels, 3, 1, 1, bias=True) # 512x512 --> 512x512
        )

    def encode(self, x):
        z = self.encoder(x)
        print(z.shape)

    def decode(self, z):
        x = self.decoder(z)
        print(x.shape)

    def reparameterize(self, mu, logvar, just_mean=False):
        if just_mean:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(logvar)
            return mu + eps * std

    def update_device(self, device):
        self.device = update_device
        
    # got the kld from the disentangled sequential autoencoder repo
    def get_kld(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    
    def vae_loss(self, x, x_pred, mu, logvar):
        recon = self.recon_loss(x_pred, x).sum((1, 2, 3))
        kld = self.get_kld(mu, logvar)
        return recon, kld

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        return self.decode(z)

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encode(x)
        encoded = encoded.view(batch_size, -1)
        mu, logvar = self.fc1(encoded), self.fc2(encoded)
        z = self.reparameterize(mu, logvar)
        z = z.reshape(batch_size, self.z_channels, self.z_h, self.z_w)
        x_pred = self.decode(z)
        recon, kld = self.vae_loss(x, x_pred, mu, lovar)
        return recon, kld