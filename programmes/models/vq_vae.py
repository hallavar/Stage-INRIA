import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence, Categorical
from .types_ import *

class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.block = nn.Sequential(
          nn.ReLU(True),
          nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(hidden_channels),
          nn.ReLU(True),
          nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class VQ(nn.Module):
    
    def __init__(self, num_embeddings, embedding_size, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.embedding.weight.data.uniform_(-1. / num_embeddings, 1. / num_embeddings)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # from BCHW to BHWC
        x_flat = x.view(-1, self.embedding_size)

        w = self.embedding.weight
        distances = torch.sum(x_flat ** 2, dim=1, keepdim=True) + torch.sum(w ** 2, dim=1) - 2 * (x_flat @ w.T)
        indices_flat = torch.argmin(distances, dim=1, keepdim=True)
        quantized_flat = self.embed(indices_flat)

        quantized = quantized_flat.view(x.shape)
        indices = indices_flat.view(*x.shape[:3]).unsqueeze(dim=1)  # BHW to BCHW


        if self.training:
            e_latent_loss = F.mse_loss(quantized.detach(), x)
            q_latent_loss = F.mse_loss(quantized, x.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss

            quantized = x + (quantized - x).detach()
        else:
            loss = 0.

        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # from BHWC to BCHW

        return quantized, indices, loss

    def embed(self, indices):
        quantized = self.embedding(indices)
        return quantized
        
class VQVAE(nn.Module):
    
    def __init__(self, in_channels, num_embeddings, embedding_size=32, res_hidden_channels=32, commitment_cost=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        
        h = embedding_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(h),
            ResBlock(h, res_hidden_channels),
            ResBlock(h, res_hidden_channels),
            ResBlock(h, res_hidden_channels)
        )
        
        self.vq = VQ(num_embeddings, embedding_size, commitment_cost)
        
        self.decoder = nn.Sequential(
            ResBlock(h, res_hidden_channels),
            ResBlock(h, res_hidden_channels),
            ResBlock(h, res_hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h, h, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h, in_channels, kernel_size=4, stride=2, padding=1)
        )
       
    def forward(self, x):
        z = self.encode(x)
        quantized, indices, vq_loss = self.quantize(z)
        x_recon = self.decode(quantized)
        return x_recon, quantized, indices, vq_loss
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def quantize(self, z):
        quantized, indices, vq_loss = self.vq(z)
        return quantized, indices, vq_loss
    
    def decode(self, quantized):
        x_recon = self.decoder(quantized)
        return x_recon
    
    def embed(self, indices):
        return self.vq.embed(indices)


