import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


def make_group_norm(num_channels: int, max_groups: int = 32):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = make_group_norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = make_group_norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = make_group_norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.cbam = CBAM(planes * self.expansion)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.cbam(out)
        out = self.relu(out)
        return out


class ResNetCBAM(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = make_group_norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.fuse_conv = nn.Conv2d(512 + 1024 + 2048, 2048, kernel_size=1, bias=False)
        self.fuse_bn = make_group_norm(2048)
        self.fuse_relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                make_group_norm(planes * block.expansion),
            )
        layers = []
        first_dilation = 1 if dilation == 1 else dilation // 2
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=first_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x, return_intermediate=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        target_size = x4.size()[2:]
        x2_up = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x3_up = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        fused = self.fuse_relu(self.fuse_bn(self.fuse_conv(torch.cat([x4, x3_up, x2_up], dim=1))))

        if return_intermediate:
            return fused, [x2, x3, x4]
        return fused


def resnet50_cbam(pretrained_path=None, **kwargs):
    model = ResNetCBAM(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained_path:
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        except TypeError:
            state_dict = torch.load(pretrained_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    return model


class ResNet50Plain(nn.Module):
    def __init__(self, pretrained_path=None, output_stride=8):
        super().__init__()
        base = resnet50(weights=None)
        if pretrained_path:
            try:
                sd = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            except TypeError:
                sd = torch.load(pretrained_path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            base.load_state_dict({k: v for k, v in sd.items() if k in base.state_dict()}, strict=False)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        if output_stride == 8:
            if self.layer3[0].downsample is not None:
                self.layer3[0].downsample[0].stride = (1, 1)
            for m in self.layer3.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                    m.stride = (1, 1)
                    m.dilation = (2, 2)
                    m.padding = (2, 2)

            if self.layer4[0].downsample is not None:
                self.layer4[0].downsample[0].stride = (1, 1)
            for m in self.layer4.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                    m.stride = (1, 1)
                    m.dilation = (4, 4)
                    m.padding = (4, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet50_plain(pretrained_path=None, output_stride=8, **kwargs):
    return ResNet50Plain(pretrained_path=pretrained_path, output_stride=output_stride)
