import torch
from torch import nn


class PixelShuffle3d(nn.Module):
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super(PixelShuffle3d, self).__init__()
        self.scale = scale

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        out_channel = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        x = x.contiguous().view(batch_size, out_channel, self.scale, self.scale, self.scale, in_depth, in_height,
                                in_width)

        output = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, out_channel, out_depth, out_height, out_width)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2, use_attn=True):
        super(UpsampleBlock, self).__init__()
        nf = in_channels
        layers = []
        for _ in range(scale_factor // 2):
            if use_attn:
                layers += [
                    SpatialAttention(),
                    nn.Conv3d(nf, nf * (2 ** 3), kernel_size=1, stride=1, padding=0),
                    PixelShuffle3d(2),
                    nn.Conv3d(nf, nf, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            else:
                layers += [
                    nn.Conv3d(nf, nf * (2 ** 3), kernel_size=1, stride=1, padding=0),
                    PixelShuffle3d(2),
                    nn.Conv3d(nf, nf, kernel_size=3, padding=1),
                    nn.GELU(),
                ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        out = x
        for module in self.layers:
            if isinstance(module, ChannelAttention) or isinstance(module, SpatialAttention):
                out = module(out) * out + out
            else:
                out = module(out)

        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2, use_attn=True):
        super(DownsampleBlock, self).__init__()
        nf = in_channels
        layers = []
        for _ in range(scale_factor // 2):
            if use_attn:
                layers += [
                    SpatialAttention(),
                    nn.Conv3d(nf, nf, kernel_size=4, stride=2, padding=1),

                    nn.GELU()
                ]
            else:
                layers += [
                    nn.Conv3d(nf, nf, kernel_size=4, stride=2, padding=1),
                    nn.GELU()
                ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        out = x
        for module in self.layers:
            if isinstance(module, ChannelAttention) or isinstance(module, SpatialAttention):
                out = module(out) * out
            else:
                out = module(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# VGG19
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            #
            # nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, num_labels)
        # )
        self.init_weights()

    def init_weights(self):
        # set weight for perceptual loss
        # according to Generic Perceptual Loss for Modeling Structured Output Dependencies
        # https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Generic_Perceptual_Loss_for_Modeling_Structured_Output_Dependencies_CVPR_2021_paper.pdf#:~:text=Thus%2C%20a%20generic%20perceptual%20loss%20for%20structured%20output,a%20wider%20range%20of%20structured%20output%20learning%20tasks.
        for l, layer in enumerate(self.features, 1):
            if isinstance(layer, nn.Conv3d):
                # nn.init.normal_(layer.weight, mean=0, std=0.01 ** 2)
                n = l * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2]
                nn.init.normal_(layer.weight, mean=0, std=(2 / n) ** 0.5)
                nn.init.constant_(layer.bias, val=0)

            # elif isinstance(layer, nn.Linear):
            #     nn.init.normal_(layer.weight, mean=0, std=0.01 ** 2)
            #     nn.init.constant_(layer.bias, val=0)

    def forward(self, x):
        x = self.features(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.classifier(x)
        return x


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)  # planes是feature map的通道个数
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x  # 广播机制
        x = self.sa(x) * x  # 广播机制
        return x


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, planes):
        super(SimplifiedChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv(self.pool(x))
        return out


class SCBAM(nn.Module):
    def __init__(self, planes):
        super(SCBAM, self).__init__()
        self.ca = SimplifiedChannelAttention(planes)  # planes是feature map的通道个数
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x  # 广播机制
        x = self.sa(x) * x  # 广播机制
        return x


class MultiScaleBlock(nn.Module):
    def __init__(self, nf):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=nf, out_channels=nf // 2, kernel_size=3, padding=1),
                                   nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=nf, out_channels=nf // 2, kernel_size=5, padding=2),
                                   nn.GELU())
        self.init_weight()

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        # out = out * self.ca(out)
        return out

    def init_weight(self):
        # nn.init.xavier_normal_(param)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class BaseBlock(nn.Module):
    def __init__(self, nf, use_attn=True):
        super(BaseBlock, self).__init__()
        self.multi_scale_block = MultiScaleBlock(nf)
        self.use_attn = use_attn
        if use_attn:
            self.scbam = SCBAM(nf)
            # self.cbam = CBAM(nf)
            # self.ca = ChannelAttention(nf)
        self.act1 = nn.GELU()

    def forward(self, x):
        out = self.multi_scale_block(x)

        if self.use_attn:
            out = self.scbam(out)
            # out = self.cbam(out)
            # out = self.ca(out) * out
        out = self.act1(out)
        return out
