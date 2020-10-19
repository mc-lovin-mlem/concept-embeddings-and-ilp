"""Loader of modified VGG16 for finetuning on picasso dataset."""
from typing import Sequence, Dict

import torch
from torchvision.models import VGG, vgg16

from ..defaults import HIDDEN_DIM, NUM_CLASSES

VGG16_FINETUNE_LAYERS: Sequence[str] = (
    'features.28', 'features.26',
    'classifier.0', 'classifier.3', 'classifier.6'
)
"""Layers to finetune (layer names from model.named_modules()) for modified VGG16."""


def modified_vgg16(state_dict: Dict[str, torch.Tensor] = None, *,
                   pretrained: bool = False,
                   num_classes: int = NUM_CLASSES, hidden_dim: int = HIDDEN_DIM) -> VGG:
    """Modify a VGG16 model to have num_classes output classes.
    A VGG16 instance is created (initialized according to pretrained) and modified as follows:
    The last before final linear dense layers is exchanged for one with hidden_dim units in- and
    output, and the number of output classes is set to num_classes.

    :param pretrained: whether to initialize the model with the pretrained VGG16 weights where
        applicable; overridden by state_dict
    :param state_dict: state dict with which to initialize parameters
    :param num_classes: number of output classes of the modified model (no sigmoid applied)
    :param hidden_dim: in- and output dimension of the second dense layer
    :return: the modified VGG instance; all non-modified layers are initialized with the
        pretrained weights if pretrained is True
    """
    vgg: VGG = vgg16(pretrained=pretrained)
    # Add fine-tuning/transfer learning modules
    vgg.classifier[0] = torch.nn.Linear(7 * 7 * 512, hidden_dim)
    vgg.classifier[3] = torch.nn.Linear(hidden_dim, hidden_dim)
    vgg.classifier[6] = torch.nn.Linear(hidden_dim, num_classes)
    if state_dict is not None:
        vgg.load_state_dict(state_dict)
    return vgg
