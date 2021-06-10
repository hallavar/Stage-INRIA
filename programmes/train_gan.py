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

from data_extraction import DataGenerator6

pth='/srv/tempdd/aschoen/preprocessed/'

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

train_dataset=DataGenerator6(pth, labels, limit, split, 'train', img_shape, 2, 16)
validation_dataset=DataGenerator6(pth, labels, limit, split, 'test', img_shape, 2, 16)

train_loader=DataLoader(train_dataset, train_batch_size, shuffle=True)
test_loader=DataLoader(validation_dataset, validation_batch_size, shuffle=True)

latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = img_shape[0] // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(100, 128, 4, 1, 0, bias=False),
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
    
optimizer_G = torch.optim.Adam(generator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = 0.0002)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(train_loader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0)).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0)).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
        z = torch.randn(imgs.size()[0], 100, 1, 1, device=device)

        # Generate a batch of images
        gen_imgs = generator(z)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
        )

torch.save(generator.state_dict(),'GAN_pkt_generator.pt')
torch.save(discriminator.state_dict(),'GAN_pkt_discriminator.pt')
