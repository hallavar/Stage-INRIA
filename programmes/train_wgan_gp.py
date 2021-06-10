import os
import numpy as np
import glob
import torch
import torchvision
from torch import nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions import kl_divergence, Categorical
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from data_extraction import DataGenerator_UDP

pth='/srv/tempdd/aschoen/preprocessed_UDP/'

img_shape = (28,28)
device='cuda'

cuda = True if torch.cuda.is_available() else False

train_batch_size=2018
validation_batch_size=1024

labels = ['Benign', 'Malware']
limit = 1000
limit = 924170
split=4/5

n_epochs = 200

train_dataset=DataGenerator_UDP(pth, labels, limit, img_shape, 2, 16)
validation_dataset=DataGenerator_UDP(pth, labels, limit, img_shape, 2, 16)

train_loader=DataLoader(train_dataset, train_batch_size, shuffle=True)
test_loader=DataLoader(validation_dataset, validation_batch_size, shuffle=True)

latent_dim = 64 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
  
            nn.ConvTranspose2d(128, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
  
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
  
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.conv_blocks(z)
        return img
        
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()    
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
                
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
                
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
                
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        return out.view(-1, 1).squeeze(1)
        
adversarial_loss = nn.BCELoss()

generator = Generator()
discriminator = Discriminator()


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    
optimizer_G = torch.optim.Adam(generator.parameters(), lr = 0.001, betas=(0.5, 0.999), weight_decay = 2.5e-5)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = 0.001, weight_decay = 2.5e-5)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

lambda_gp = 4
n_critic = 5

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0]).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
batches_done = 0
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(train_loader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.size()[0], latent_dim, 1, 1, device=device)

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        fake_img=fake_imgs.squeeze()
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
            )

            batches_done += n_critic

torch.save(generator.state_dict(),'WGAN_GP_pkt_generator.pt')
torch.save(discriminator.state_dict(),'WGAN_GP_pkt_discriminator.pt')

