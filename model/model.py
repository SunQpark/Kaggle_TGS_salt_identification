import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from base import BaseModel


def make_block(in_ch, out_ch, dropout=0.1, residual=False, upsample=False, activation='relu', last_activation=True):
    if activation == 'relu':
        activ = nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        activ = nn.LeakyReLU(0.2, inplace=True),
    elif activation == 'elu':
        activ = nn.ELU(inplace=True)
    
    if residual:
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            nn.Dropout2d(p=dropout, inplace=False),
            nn.BatchNorm2d(out_ch),
            activ,

            ResidualBlock(out_ch, out_ch, activ)
        ]
    else:
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            nn.Dropout2d(p=dropout, inplace=False),
            nn.BatchNorm2d(out_ch),
            activ,

            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
            nn.Dropout2d(p=dropout, inplace=False),
            nn.BatchNorm2d(out_ch),
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
            nn.Conv2d(ch, ch, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(ch),
            activation)

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
            make_block(in_ch,   n_fts,   residual=residual),
            make_block(n_fts,   n_fts*2, residual=residual),
            make_block(n_fts*2, n_fts*4, residual=residual),
            make_block(n_fts*4, n_fts*8, residual=residual),
        )

        self.bottleneck = nn.Sequential(
            make_block(n_fts*8,  n_fts*16, residual=residual),
            make_block(n_fts*16, n_fts*16, residual=residual, upsample=True)
        )

        self.decoder = nn.Sequential(
            make_block(n_fts*16, n_fts*8, residual=residual, upsample=True),
            make_block(n_fts*8,  n_fts*4, residual=residual, upsample=True),
            make_block(n_fts*4,  n_fts*2, residual=residual, upsample=True),
            make_block(n_fts*2,  n_fts,   residual=residual, upsample=False),
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


class EmptyFilter(nn.Module):
    def __init__(self):
        super(EmptyFilter, self).__init__()
        resnet = resnet18(pretrained=False)
        self.layers = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(512, 1)
        # print(self.resnet)
        
    def forward(self, x_input):
        x = self.layers(x_input)
        output = self.fc(x.view(-1, 512))
        return F.sigmoid(output)


class UnetWithEmptyClassifier(nn.Module):
    def __init__(self, n_fts):
        super(UnetWithEmptyClassifier, self).__init__()
        self.clsfier = EmptyFilter()
        self.unet = Unet(n_fts=n_fts)

    def forward(self, x_input):
        # b, c, w, h = x_input.shape
        emptiness = self.clsfier(x_input)
        nemp_idx = (emptiness > 0.5).long()
        # shield for empty batch
        if torch.max(nemp_idx) == 0:
            nemp_idx[0] = 1
        nemp_img = torch.masked_select(x_input, nemp_idx)
        nemp_out = self.unet(nemp_img)
        return emptiness, nemp_idx, nemp_out



if __name__ == '__main__':
    dummy_input = torch.randn(4, 3, 112, 112)
    model = Unet()
    # model = UnetWithResnetEncoder()
    print(model)
    out = model(dummy_input)
    print(out.shape)
    

