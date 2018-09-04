from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, n_features=32):
        super(Unet, self).__init__()
        self.encoder = nn.Sequential(
            self._make_block(in_ch,        n_features),
            self._make_block(n_features  , n_features*2),
            self._make_block(n_features*2, n_features*4),
            self._make_block(n_features*4, n_features*8),
        )

        self.bottleneck = nn.Sequential(
            self._make_block(n_features*8,  n_features*16),
            self._make_block(n_features*16, n_features*16),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(n_features*16, n_features*8, 3, 1, padding=1),
            nn.BatchNorm2d(n_features*8),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            self._make_block(n_features*16, n_features*8, upsample=True),
            self._make_block(n_features*8, n_features*4, upsample=True),
            self._make_block(n_features*4, n_features*2, upsample=True),
            self._make_block(n_features*2, n_features  , upsample=False),
        )
        self.output = nn.Conv2d(n_features, out_ch, 1, 1)


    def _make_block(self, in_ch, out_ch, upsample=False, activation='relu', last_activation=True):
        if activation == 'relu':
            activ = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activ = nn.LeakyReLU(0.2, inplace=True),
        elif activation == 'elu':
            activ = nn.ELU(inplace=True)

        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            nn.BatchNorm2d(out_ch),
            activ,

            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
            nn.BatchNorm2d(out_ch),
        ]
        if upsample:
            layers += [activ, 
                       nn.Upsample(scale_factor=2),
                       nn.Conv2d(out_ch, out_ch // 2, 3, 1, padding=1),
                       nn.BatchNorm2d(out_ch // 2)]
        if last_activation:
            layers += [activ]
        return nn.Sequential(*layers)


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


if __name__ == '__main__':
    dummy_input = torch.randn(4, 3, 128, 128)
    model = Unet()
    out = model(dummy_input)
    print(out.shape)
