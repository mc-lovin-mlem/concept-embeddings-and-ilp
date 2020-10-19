"""Loader of modified MobileNetV2 for finetuning on picasso dataset."""

from typing import Sequence, Dict

import torch
from torchvision.models import MobileNetV2, mobilenet_v2

from ..defaults import NUM_CLASSES

MOBILENETV2_FINETUNE_LAYERS: Sequence[str] = (
    'features.17',
    'features.18',
    'classifier',
)
"""Layers to finetune (layer names from model.named_modules()) for modified ResNeXt."""


def modified_mobilenetv2(state_dict: Dict[str, torch.Tensor] = None, *,
                         pretrained: bool = False,
                         num_classes: int = NUM_CLASSES) -> MobileNetV2:
    """Modify a ResNeXt50 model to have num_classes output classes.
    A MobileNetV2 instance is created (initialized according to pretrained) and modified as follows:
    The last (and only) fully connected layer is replaced by one with num_classes output classes.

    :param pretrained: whether to initialize the model with the pretrained VGG16 weights where
        applicable; overridden by state_dict
    :param state_dict: state dict with which to initialize parameters
    :param num_classes: number of output classes of the modified model (no sigmoid applied)
    :return: the modified VGG instance; all non-modified layers are initialized with the
        pretrained weights if pretrained is True
    """
    mobilenetv2: MobileNetV2 = mobilenet_v2(pretrained=pretrained)
    # Add fine-tuning/transfer learning modules
    mobilenetv2.classifier[1] = torch.nn.Linear(1280, num_classes)
    if state_dict is not None:
        mobilenetv2.load_state_dict(state_dict)
    return mobilenetv2
