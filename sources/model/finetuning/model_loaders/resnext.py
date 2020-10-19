"""Loader of modified ResNeXt50 for finetuning on picasso dataset."""

from typing import Sequence, Dict

import torch
from torchvision.models import ResNet, resnext50_32x4d

from ..defaults import NUM_CLASSES

RESNEXT_FINETUNE_LAYERS: Sequence[str] = (
    'layer4.1', 'layer4.2', 'fc'
)
"""Layers to finetune (layer names from model.named_modules()) for modified ResNeXt."""


def modified_resnext50(state_dict: Dict[str, torch.Tensor] = None, *,
                       pretrained: bool = False,
                       num_classes: int = NUM_CLASSES) -> ResNet:
    """Modify a ResNeXt50 model to have num_classes output classes.
    A ResNeXt50 instance is created (initialized according to pretrained) and modified as follows:
    The last (and only) fully connected layer is replaced by one with num_classes output classes.

    :param pretrained: whether to initialize the model with the pretrained VGG16 weights where
        applicable; overridden by state_dict
    :param state_dict: state dict with which to initialize parameters
    :param num_classes: number of output classes of the modified model (no sigmoid applied)
    :return: the modified VGG instance; all non-modified layers are initialized with the
        pretrained weights if pretrained is True
    """
    resnext50: ResNet = resnext50_32x4d(pretrained=pretrained)
    # Add fine-tuning/transfer learning modules
    resnext50.fc = torch.nn.Linear(2 * 2 * 512, num_classes)
    if state_dict is not None:
        resnext50.load_state_dict(state_dict)
    return resnext50
