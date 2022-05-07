import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['get_resnet', 'BasicBlock']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, out_keys=None, in_channels=3, **kwargs):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.out_keys = out_keys
        self.num_classes = num_classes
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        if 'block5' in self.out_keys:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        if self.num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        endpoints = dict()
        endpoints['block0'] = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        endpoints['block1'] = x
        x = self.maxpool(x)
        x = self.layer1(x)
        endpoints['block2'] = x
        x = self.layer2(x)
        endpoints['block3'] = x
        x = self.layer3(x)
        endpoints['block4'] = x
        if 'block5' in self.out_keys:
            x = self.layer4(x)
            endpoints['block5'] = x

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        if self.out_keys is not None:
            endpoints = {key: endpoints[key] for key in self.out_keys}
        return x, endpoints


def _resnet(arch, block, layers, pretrained, progress, num_classes=1000, in_channels=3, out_keys=None, **kwargs):
    model = ResNet(block, layers, num_classes, out_keys=out_keys, in_channels=in_channels, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        if in_channels != 3:
            keys = state_dict.keys()
            keys = [x for x in keys if 'conv1.weight' in x]
            for key in keys:
                del state_dict[key]
        if num_classes !=1000:
            keys = state_dict.keys()
            keys = [x for x in keys if 'fc' in x]
            for key in keys:
                del state_dict[key]
        if 'block5' not in out_keys:
            keys = state_dict.keys()
            keys = [x for x in keys if 'layer4' in x]
            for key in keys:
                del state_dict[key]
        model.load_state_dict(state_dict)
        print('load resnet model...')
        
    return model


def _resnet18(name='resnet18', pretrained=True, progress=True, num_classes=1000, out_keys=None, **kwargs):
    r"""ResNet-18 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
    return _resnet(name, BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   num_classes=num_classes, out_keys=out_keys, **kwargs)

def _resnet50(name='resnet50',pretrained=False, progress=True,num_classes=1000,out_keys=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(name, Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   num_classes=num_classes,out_keys=out_keys,
                   **kwargs)


def _resnet101(name='resnet101',pretrained=False, progress=True, num_classes=1000,out_keys=None,**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(name, Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   num_classes=num_classes, out_keys=out_keys,
                   **kwargs)


def _resnet152(name='resnet152',pretrained=False, progress=True,num_classes=1000,out_keys=None,**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(name, Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   num_classes=num_classes, out_keys=out_keys,
                   **kwargs)


def get_resnet(model_name='resnet50', pretrained=True, progress=True, num_classes=1000, out_keys=None, in_channels=3, **kwargs):
    '''
    Get resnet model with name.
    :param name: resnet model name, optional values:[resnet18, reset50, resnet101, resnet152]
    :param pretrained: If True, returns a model pre-trained on ImageNet
    '''

    if pretrained and num_classes != 1000:
        print('warning: num_class is not equal to 1000, which will cause some parameters to fail to load!')
    if pretrained and in_channels != 3:
        print('warning: in_channels is not equal to 3, which will cause some parameters to fail to load!')

    if model_name == 'resnet18':
        return _resnet18(name=model_name, pretrained=pretrained, progress=progress,
                         num_classes=num_classes, out_keys=out_keys, in_channels=in_channels, **kwargs)
    elif model_name == 'resnet50':
        return _resnet50(name=model_name, pretrained=pretrained, progress=progress,
                         num_classes=num_classes, out_keys=out_keys, in_channels=in_channels, **kwargs)
    elif model_name == 'resnet101':
        return _resnet101(name=model_name, pretrained=pretrained, progress=progress,
                          num_classes=num_classes, out_keys=out_keys, in_channels=in_channels, **kwargs)
    elif model_name == 'resnet152':
        return _resnet152(name=model_name, pretrained=pretrained, progress=progress,
                          num_classes=num_classes, out_keys=out_keys, in_channels=in_channels, **kwargs)
    else:
        raise NotImplementedError(r'''{0} is not an available values. \
                                  Please choose one of the available values in
                                   [resnet18, reset50, resnet101, resnet152]'''.format(name))


if __name__ == '__main__':
    model = get_resnet('resnet18', pretrained=True, num_classes=None, in_channels=3, out_keys=['block4'])
    x = torch.rand([2, 3, 256, 256])
    torch.save(model.state_dict(), 'res18nofc.pth')