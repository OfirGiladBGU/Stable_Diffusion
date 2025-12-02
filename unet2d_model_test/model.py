import torch
import torch.nn as nn

def conv_block_2d(in_channels, out_channels, kernel_size=3, padding=1, norm='batch', activation='relu', dropout=0.0):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)]
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels, affine=True))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.01, inplace=True))
    if dropout and dropout > 0.0:
        layers.append(nn.Dropout2d(p=dropout))
    return nn.Sequential(*layers)

class ResidualConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm='batch', activation='relu', dropout=0.0):
        super().__init__()
        self.conv1 = conv_block_2d(in_channels, out_channels, norm=norm, activation=activation, dropout=dropout)
        self.conv2 = conv_block_2d(out_channels, out_channels, norm=norm, activation=activation, dropout=dropout)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + self.skip(x)

class Encoder2D(nn.Module):
    def __init__(self, in_channels, fmaps, depth=4, norm='batch', activation='relu', dropout=0.0, residual=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        c_in = in_channels
        for d in range(depth):
            c_out = fmaps * (2 ** d)
            block = ResidualConvBlock2D(c_in, c_out, norm=norm, activation=activation, dropout=dropout) if residual else nn.Sequential(
                conv_block_2d(c_in, c_out, norm=norm, activation=activation, dropout=dropout),
                conv_block_2d(c_out, c_out, norm=norm, activation=activation, dropout=dropout)
            )
            self.blocks.append(block)
            if d < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
            c_in = c_out
        self.out_channels = c_in
    def forward(self, x):
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            feats.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        return x, feats

class Decoder2D(nn.Module):
    def __init__(self, out_channels, enc_channels, norm='batch', activation='relu', dropout=0.0, residual=True):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        c_in = enc_channels[-1]
        for skip_c in reversed(enc_channels[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2))
            c_cat = c_in // 2 + skip_c
            block = ResidualConvBlock2D(c_cat, skip_c, norm=norm, activation=activation, dropout=dropout) if residual else nn.Sequential(
                conv_block_2d(c_cat, skip_c, norm=norm, activation=activation, dropout=dropout),
                conv_block_2d(skip_c, skip_c, norm=norm, activation=activation, dropout=dropout)
            )
            self.blocks.append(block)
            c_in = skip_c
        self.final_conv = nn.Conv2d(c_in, out_channels, kernel_size=1)
    def forward(self, x, feats):
        # feats: list of encoder outputs per level
        for up, block, skip in zip(self.upconvs, self.blocks, reversed(feats[:-1])):
            x = up(x)
            # pad if needed
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return self.final_conv(x)

class UNet2D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 fmaps=32,
                 depth=4,
                 norm='batch',
                 activation='relu',
                 dropout=0.0,
                 residual=True,
                 final_activation='sigmoid'):
        super().__init__()
        self.encoder = Encoder2D(in_channels, fmaps=fmaps, depth=depth, norm=norm, activation=activation, dropout=dropout, residual=residual)
        enc_channels = [fmaps * (2 ** d) for d in range(depth)]
        self.decoder = Decoder2D(out_channels, enc_channels=enc_channels, norm=norm, activation=activation, dropout=dropout, residual=residual)
        if final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_activation == 'identity':
            self.final_act = nn.Identity()
        else:
            raise ValueError("Unsupported final_activation: use 'sigmoid' or 'identity'")
    def forward(self, x):
        bottleneck, feats = self.encoder(x)
        logits = self.decoder(bottleneck, feats)
        return self.final_act(logits)

if __name__ == "__main__":
    model = UNet2D(in_channels=1, out_channels=1, fmaps=32, depth=4, norm='batch', residual=True, final_activation='sigmoid')
    x = torch.randn(2,1,512,512)
    y = model(x)
    print(x.shape, y.shape)
