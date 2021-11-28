from .ResNet import *
from .VGGNet import *

__all__ = ['get_backbone']


def get_backbone(model_name='', pretrained=True, num_classes=None, **kwargs):
    if 'res' in model_name:
        model = get_resnet(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)

    elif 'vgg' in model_name:
        model = get_vgg(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    else:
        raise NotImplementedError
    return model

