import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast


__all__ = ['get_vgg']


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        num_classes,
        out_keys,
        output_make_layers,
        init_weights: bool = True,
        **kwargs
    ) -> None:
        super(VGG, self).__init__()
        self.stage_id = output_make_layers[0]
        self.features = output_make_layers[1]
        self.num_classes = num_classes
        self.out_keys = out_keys
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor):
        out_blocks = dict()
        stage = 0
        out_blocks['block%d' % stage] = x

        for idx, op in enumerate(self.features):
            if idx in self.stage_id:
                stage += 1
                x = op(x)
                out_blocks['block%d' % stage] = x
                continue
            x = op(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        if self.out_keys is not None:
            out_blocks = {key: out_blocks[key] for key in self.out_keys}
        return x, out_blocks

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(in_channels, out_keys, cfg: List[Union[str, int]], batch_norm: bool = False):
    layer_list = []

    idx = 0
    stage_ids = []
    for v in cfg:
        if isinstance(v, int) and v in [1, 2, 3, 4, 5]:
            if v > int(out_keys[-1].replace('block', '')):
                break
            continue
        if v == 'M':
            layer_list += [nn.MaxPool2d(kernel_size=2, stride=2)]
            stage_ids += [idx]
            idx += 1
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layer_list += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                idx += 3
            else:
                layer_list += [conv2d, nn.ReLU(inplace=True)]
                idx += 2
            in_channels = v

    return stage_ids, nn.Sequential(*layer_list)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [1, 64, 64, 'M', 2, 128, 128, 'M', 3, 256, 256, 256, 'M', 4, 512, 512, 512, 'M', 5, 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(in_channels, num_classes, out_keys, arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    stage_id, ops = make_layers(in_channels, out_keys, cfgs[cfg], batch_norm=batch_norm)
    model = VGG(num_classes, out_keys, (stage_id, ops), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        if in_channels != 3:
            keys = state_dict.keys()
            keys = [x for x in keys if 'features.0.' in x]
            for key in keys:
                del state_dict[key]
        if num_classes != 1000:
            keys = state_dict.keys()
            keys = [x for x in keys if 'classifier' in x]
            for key in keys:
                del state_dict[key]
        if 'block5' not in out_keys:
            keys = list(state_dict.keys())
            for key in keys:
                key_layer_id = int(key.split('.')[1])
                if key_layer_id >= stage_id[-1]:
                    del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(in_channels, num_classes, out_keys, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(in_channels, num_classes, out_keys,'vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


def get_vgg(name='vgg16_bn', pretrained=True, progress=True, num_classes=None, out_keys=None, in_channels=3, **kwargs):

    if pretrained and num_classes != 1000:
        print('warning: num_class is not equal to 1000, which will cause some parameters to fail to load!')
    if pretrained and in_channels != 3:
        print('warning: in_channels is not equal to 3, which will cause some parameters to fail to load!')

    if name == 'vgg16_bn':
        return vgg16_bn(in_channels=in_channels, num_classes=num_classes,
                        out_keys=out_keys, pretrained=pretrained, progress=progress, **kwargs)

    elif name == 'resnet50':
        return _resnet50(name=name, pretrained=pretrained, progress=progress,
                         num_classes=num_classes, out_keys=out_keys, in_channels=in_channels, **kwargs)
    elif name == 'resnet101':
        return _resnet101(name=name, pretrained=pretrained, progress=progress,
                          num_classes=num_classes, out_keys=out_keys, in_channels=in_channels, **kwargs)
    elif name == 'resnet152':
        return _resnet152(name=name, pretrained=pretrained, progress=progress,
                          num_classes=num_classes, out_keys=out_keys, in_channels=in_channels, **kwargs)
    else:
        raise NotImplementedError(r'''{0} is not an available values. \
                                  Please choose one of the available values in
                                   [resnet18, reset50, resnet101, resnet152]'''.format(name))


if __name__ == '__main__':
    model = get_vgg('vgg16_bn', pretrained=True, num_classes=None, in_channels=4, out_keys=['block3'])
    x = torch.rand([2, 3, 512, 512])
    x = model(x)
    torch.save(model.state_dict(), '../../vgg16bns4.pth')
