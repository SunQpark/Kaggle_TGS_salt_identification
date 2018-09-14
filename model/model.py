import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from base import BaseModel


def make_block(in_ch, out_ch, dropout=0.5, residual=False, upsample=False, activation='relu', last_activation=True):
    if activation == 'relu':
        activ = nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        activ = nn.LeakyReLU(0.2, inplace=True),
    elif activation == 'elu':
        activ = nn.ELU(inplace=True)
    
    if residual:
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            nn.BatchNorm2d(out_ch),
            # activ,
            ResidualBlock(out_ch, out_ch//4, activ),
            ResidualBlock(out_ch, out_ch//4, activ)
        ]
    else:
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            nn.BatchNorm2d(out_ch),
            activ,

            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
            nn.BatchNorm2d(out_ch)
        ]

    if upsample:
        layers += [
            activ, 
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_ch, out_ch // 2, 3, 1, padding=1),
            nn.BatchNorm2d(out_ch // 2)
        ]

    if last_activation:
        layers += [activ]

    if dropout != 0.0:
        layers += [nn.Dropout2d(p=dropout, inplace=False)]

    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, ch, activation, stride=1, downsample=None, groups=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, ch, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(ch),
            activation)

        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(ch),
            activation)

        self.conv3 = nn.Sequential(
            nn.Conv2d(ch, in_ch, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(in_ch))

        self.downsample = downsample

    def forward(self, x_input):
        skip = x_input
        x = self.conv1(x_input)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            skip = self.downsample(skip)
        return x + skip


class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, n_fts=32, residual=False):
        super(Unet, self).__init__()
        self.encoder = nn.Sequential(
            make_block(in_ch,   n_fts,   residual=residual, dropout=0.25),
            make_block(n_fts,   n_fts*2, residual=residual, dropout=0.5),
            make_block(n_fts*2, n_fts*4, residual=residual, dropout=0.5),
            make_block(n_fts*4, n_fts*8, residual=residual, dropout=0.5),
        )

        self.bottleneck = nn.Sequential(
            make_block(n_fts*8,  n_fts*16, residual=residual, dropout=0.5, upsample=True),
            # make_block(n_fts*16, n_fts*16, residual=residual, dropout=0.5),
            # make_block(n_fts*16, n_fts*16, residual=residual, dropout=0.5)
        )

        self.decoder = nn.Sequential(
            make_block(n_fts*16, n_fts*8, residual=residual, dropout=0.5,  upsample=True),
            make_block(n_fts*8,  n_fts*4, residual=residual, dropout=0.5,  upsample=True),
            make_block(n_fts*4,  n_fts*2, residual=residual, dropout=0.5,  upsample=True),
            make_block(n_fts*2,  n_fts,   residual=residual, dropout=0.25, upsample=False),
        )
        self.output = nn.Conv2d(n_fts, out_ch, 1, 1)
    
    def forward(self, x):
        skip = []
        for down_block in self.encoder.children():
            x = down_block(x)
            skip.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for up_block in self.decoder.children():
            x = torch.cat([skip.pop(), x], dim=1)
            x = up_block(x)

        x = self.output(x)
        x = (F.tanh(x) + 1)/2
        return x


class ResnetUnet(nn.Module):
    def __init__(self, n_fts=32, out_ch=1):
        super(ResnetUnet, self).__init__()
        resnet = resnet34(pretrained=True)

        self.layers = nn.Sequential(*list(resnet.children())[:-2])
        for i, l in enumerate(self.layers.children()):
            if i <= 5:
                l.requires_grad = False

        self.skip = []
        def hook(module, input, output): self.skip.append(output)
        for l in list(self.layers.children())[4:]:
            l.register_forward_hook(hook)

        self.decoder = nn.Sequential(
            make_block(n_fts*32, n_fts*16, upsample=True),
            make_block(n_fts*16, n_fts*8,  upsample=True),
            make_block(n_fts*8,  n_fts*4,  upsample=True),
            make_block(n_fts*4,  n_fts*2,  upsample=True),
        )
        self.empty_filter = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((-1, 1)),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.output = nn.Sequential(
            make_block(n_fts,  n_fts,  upsample=True),
            nn.Conv2d(n_fts//2, out_ch, 1, 1),
        )

    def forward(self, x_input):
        self.skip = []
        x = self.layers(x_input)

        for up_block in self.decoder.children():
            x = torch.cat([self.skip.pop(), x], dim=1)
            x = up_block(x)
        
        x = self.output(x)
        output = (F.tanh(x) + 1)/2
        return output



if __name__ == '__main__':
    dummy_input = torch.randn(4, 3, 128, 128)
    model = ResnetUnet()
    # print(model)
    cls_out, seg_out = model(dummy_input)
    print(cls_out.shape)
    print(seg_out.shape)
    

