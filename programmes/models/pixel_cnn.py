import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence, Categorical
from .types_ import *

class GatedActivation(nn.Module):
  
    def __init__(self):
        super().__init__()
  
    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    
    def __init__(self, mask_type, hidden_channels, kernel_size, residual, num_classes):
        super().__init__()
        self.mask_type = mask_type
        self.residual = residual

        h = hidden_channels

        self.class_cond_embedding = nn.Embedding(num_classes, h * 2)

        self.vert_stack = nn.Conv2d(
            h, h * 2, kernel_size=(kernel_size // 2 + 1, kernel_size), stride=1,
            padding=(kernel_size // 2, kernel_size // 2)
        )
        self.vert_to_horiz = nn.Conv2d(h * 2, h * 2, kernel_size=1, stride=1, padding=0)
        self.horiz_stack = nn.Conv2d(h, h * 2, kernel_size=(1, kernel_size // 2 + 1), stride=1,
                                      padding=(0, kernel_size // 2))
        self.horiz_resid = nn.Conv2d(h, h, kernel_size=1, stride=1, padding=0)

        self.gate = GatedActivation()
  
    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()
        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    
    def __init__(self, num_classes, in_channels, hidden_channels, output_channels, num_layers):
        super().__init__()
        self.num_classes = num_classes

        self.embedding = nn.Embedding(in_channels, hidden_channels)

        self.mask_a = GatedMaskedConv2d('A', hidden_channels, kernel_size=7,
                                        residual=False, num_classes=num_classes)
        self.mask_bs = nn.ModuleList([
            GatedMaskedConv2d('B', hidden_channels, kernel_size=3,
                              residual=True, num_classes=num_classes) for _ in range(num_layers - 1)])

        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, output_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, y):
        x = self.embedding(x.view(-1)).view(x.shape + (-1,))  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x_v, x_h = self.mask_a(x, x, y)
        for mask_b in self.mask_bs:
            x_v, x_h = mask_b(x_v, x_h, y)

        return self.output_conv(x_h)

    def sample(self, batch_size, shape, label=None):
        self.eval()

        x = torch.zeros((batch_size, *shape), dtype=torch.long, device=device)
        if label is None:
            label = torch.randint(self.num_classes, (batch_size,), dtype=torch.long, device=device)

        with torch.no_grad():
            for i in range(shape[0]):
                for j in range(shape[1]):
                    logits = self.forward(x, label)
                    dist = Categorical(logits=logits[:, :, i, j])
                    x[:, i, j] = dist.sample()
        return x
        
def convert_dataset(model, dataset, batch_size, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    all_indices = []
    all_y = []
    model.eval()
    for x, y in dataloader:
        x = x.to(device)
        with torch.no_grad():
            z = model.encode(x)
            quantized, indices, _ = model.quantize(z)
        all_indices.append(indices)
        all_y.append(y)
    
    indices = torch.cat(all_indices, dim=0)
    y = torch.cat(all_y, dim=0)
    return torch.utils.data.TensorDataset(indices, y)
