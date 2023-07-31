import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )


def deconv3x3(in_channels, out_channels, stride):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        output_padding=1,
        bias=False,
    )


def maxpool2x2(stride):
    return nn.MaxPool2d(kernel_size=2, stride=stride, padding=0)


def relu(inplace=True):
    return nn.ReLU(inplace=inplace)


def bn(num_features):
    return nn.BatchNorm2d(num_features=num_features)


class ConvBlock(nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, max_pool=False):
        super(ConvBlock, self).__init__()

        layers = []
        for i in range(num_layer):
            _in_channels = in_channels if i == 0 else out_channels
            layers.append(conv3x3(_in_channels, out_channels))
            layers.append(bn(out_channels))
            layers.append(relu())

        if max_pool:
            layers.append(maxpool2x2(stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = num_filters[4]

        # Block 1-4
        _in_channels = self.in_channels
        self.block1 = ConvBlock(
            num_layers[0], _in_channels, num_filters[0], max_pool=True
        )
        self.block2 = ConvBlock(
            num_layers[1], num_filters[0], num_filters[1], max_pool=True
        )
        self.block3 = ConvBlock(
            num_layers[2], num_filters[1], num_filters[2], max_pool=True
        )
        self.block4 = ConvBlock(num_layers[3], num_filters[2], num_filters[3])

        # Block 5
        _in_channels = sum(num_filters[0:4])
        self.block5 = ConvBlock(num_layers[4], _in_channels, num_filters[4])

    def forward(self, x):
        N, C, H, W = x.shape

        # the first 4 blocks
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)

        # upsample and concat
        _H, _W = H // 4, W // 4
        c1_interp = F.interpolate(
            c1, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c2_interp = F.interpolate(
            c2, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c3_interp = F.interpolate(
            c3, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c4_interp = F.interpolate(
            c4, size=(_H, _W), mode="bilinear", align_corners=True
        )

        #
        c4_aggr = torch.cat((c1_interp, c2_interp, c3_interp, c4_interp), dim=1)
        c5 = self.block5(c4_aggr)

        return c5


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            deconv3x3(in_channels, 128, stride=2),
            bn(128),
            relu(),
            conv3x3(128, 128),
            bn(128),
            relu(),
            deconv3x3(128, 64, stride=2),
            bn(64),
            relu(),
            conv3x3(64, 64),
            bn(64),
            relu(),
            conv3x3(64, out_channels, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class OccupancyForecastingNetwork(nn.Module):
    def __init__(self, p_pre, p_post, n_height):
        super(OccupancyForecastingNetwork, self).__init__()

        self.p_pre = p_pre
        self.p_post = p_post
        self.n_height = n_height

        _in_channels = self.p_pre * self.n_height
        self.encoder = Encoder(_in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256])

        _out_channels = self.p_post * self.n_height
        self.linear = torch.nn.Conv2d(
            _in_channels, _out_channels, (3, 3), stride=1, padding=1, bias=True
        )
        self.decoder = Decoder(self.encoder.out_channels, _out_channels)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_occ=None, gt_occ=None, invalid_mask=None):
        # double check
        N, T_in, L, W, H = input_occ.shape
        assert T_in == self.p_pre and H == self.n_height

        # transpose axis
        input_occ = torch.movedim(input_occ, 4, 2).contiguous()
        _input = input_occ.reshape(N, -1, L, W)

        # w/ skip connection
        _output = self.linear(_input) + self.decoder(self.encoder(_input))
        output = _output.reshape(N, -1, H, L, W)
        output = torch.movedim(output, 2, 4)

        if self.training:
            valid_output = output[~invalid_mask]
            gt_occ[gt_occ > 0] = 1
            valid_gt = gt_occ[~invalid_mask]
            loss = self.criterion(valid_output, valid_gt.to(torch.float32))
            return loss
        else:
            return output
