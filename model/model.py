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
            activ,
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

def conv4x4(in_ch, out_ch, stride=1, activation=None):
    if activation is None:
        activation = nn.ReLU(inplace=True)
    return nn.Sequential(
        nn.ReflectionPad2d([1, 2, 1, 2]),
        nn.Conv2d(in_ch, out_ch, 4, stride),
        nn.BatchNorm2d(out_ch),
        activation
    )

def upconv4x4(in_ch, out_ch, stride=1, activation=None, checker_fix=True):
    if activation is None:
        activation = nn.ReLU(inplace=True)
    if checker_fix:
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d([1, 2, 1, 2]),
            nn.Conv2d(in_ch, out_ch, 4, stride),
            nn.BatchNorm2d(out_ch),
            activation
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride, padding=1),
            nn.BatchNorm2d(out_ch),
            activation
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, ch, activation=None, stride=1, groups=1):
        super(ResidualBlock, self).__init__()
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation
        assert ch % 4 == 0
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, ch//4, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(ch//4),
            self.activation)

        self.conv2 = nn.Sequential(
            nn.Conv2d(ch//4, ch//4, 3, stride, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(ch//4),
            self.activation)

        self.conv3 = nn.Sequential(
            nn.Conv2d(ch//4, ch, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(ch))
        
        if in_ch != ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, ch, 3, stride, padding=1, bias=False),
                nn.BatchNorm2d(ch)
            )
        else:
            self.downsample = None

    def forward(self, x_input):
        skip = x_input
        x = self.conv1(x_input)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            skip = self.downsample(skip)
        return self.activation(x + skip)


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
            make_block(n_fts*8,  n_fts*16, residual=residual, dropout=0.5),
            make_block(n_fts*16, n_fts*16, residual=residual, dropout=0.5),
            make_block(n_fts*16, n_fts*16, residual=residual, dropout=0.5, upsample=True)
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
        self.output = nn.Sequential(
            make_block(n_fts,  n_fts,  upsample=True),
            nn.Conv2d(n_fts//2, out_ch, 1, 1),
        )

    def forward(self, x_input):
        self.skip = []
        x = self.layers(x_input)

        assert len(self.skip) == 4
        for up_block in self.decoder.children():
            x = torch.cat([self.skip.pop(), x], dim=1)
            x = up_block(x)
        
        x = self.output(x)
        output = (F.tanh(x) + 1)/2
        return output


class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, n_fts=64):
        super(Generator, self).__init__()
        self.conv1 = conv4x4(in_ch, n_fts, stride=2)
        self.encode1 = nn.Sequential(
            conv4x4(n_fts, n_fts*2, stride=2),
            self.residual_blocks(n_fts*2, n_fts*4, 3),
        )
        self.encode2 = nn.Sequential(
            conv4x4(n_fts*4, n_fts*8, stride=2),
            self.residual_blocks(n_fts*8, n_fts*8, 3),
        )
        self.bottleneck = nn.Sequential(
            conv4x4(n_fts*8, n_fts*8, stride=1),
            self.residual_blocks(n_fts*8, n_fts*8, 3),
            upconv4x4(n_fts*8, n_fts*4, stride=2)
        )
        self.decode1 = nn.Sequential(
            self.residual_blocks(n_fts*12, n_fts*4, 3),
            upconv4x4(n_fts*4, n_fts*2),
        )
        self.decode2 = nn.Sequential(
            self.residual_blocks(n_fts*6, n_fts, 3),
            upconv4x4(n_fts, n_fts)
        )
        self.output = upconv4x4(n_fts, out_ch, activation=nn.Sigmoid())

    def residual_blocks(self, in_ch, ch, n_res):
        layers = [ResidualBlock(in_ch, ch)] + [ResidualBlock(ch, ch) for _ in range(n_res-1)]
        return nn.Sequential(*layers)

    def forward(self, x_input):
        skip = []
        x = self.conv1(x_input)
        x = self.encode1(x)
        skip.append(x)
        x = self.encode2(x)
        skip.append(x)
        x = self.bottleneck(x)
        x = torch.cat([skip.pop(), x], dim=1)
        x = self.decode1(x)
        x = torch.cat([skip.pop(), x], dim=1)
        x = self.decode2(x)
        return self.output(x)



if __name__ == '__main__':
    dummy_input = torch.randn(4, 3, 104, 104)
    model = Generator()
    # print(model)
    output = model(dummy_input)
    print(output.shape)
    

