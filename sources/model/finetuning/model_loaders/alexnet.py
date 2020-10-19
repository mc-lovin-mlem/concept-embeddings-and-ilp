"""Loader of modified alexnet for finetuning on picasso dataset."""
from typing import Sequence, Dict

import torch
from torchvision.models import AlexNet, alexnet

from ..defaults import HIDDEN_DIM, NUM_CLASSES

ALEXNET_FINETUNE_LAYERS: Sequence[str] = (
    'features.10', 'classifier.1', 'classifier.4', 'classifier.6'
)
"""Layers to finetune (layer names from model.named_modules()) for modified AlexNet."""


def modified_alexnet(state_dict: Dict[str, torch.Tensor] = None, *,
                     pretrained: bool = False,
                     num_classes: int = NUM_CLASSES, hidden_dim: int = HIDDEN_DIM) -> AlexNet:
    """Modify an AlexNet model to have num_classes output classes.
    An AlexNet instance is created (initialized according to `pretrained`) and modified as follows:
    The last before final linear dense layers is exchanged for one with hidden_dim units in- and
    output, and the number of output classes is set to num_classes.

    :param pretrained: whether to initialize the model with the pretrained VGG16 weights where
        applicable; overridden by state_dict
    :param state_dict: state dict with which to initialize parameters
    :param num_classes: number of output classes of the modified model (no sigmoid applied)
    :param hidden_dim: in- and output dimension of the second dense layer
    :return: the modified AlexNet instance; all non-modified layers are initialized with the
        pretrained weights if pretrained is True
    """
    alexn: AlexNet = alexnet(pretrained=pretrained)
    # Add fine-tuning/transfer learning modules
    alexn.classifier[1] = torch.nn.Linear(6 * 6 * 256, hidden_dim)
    alexn.classifier[4] = torch.nn.Linear(hidden_dim, hidden_dim)
    alexn.classifier[6] = torch.nn.Linear(hidden_dim, num_classes)
    if state_dict is not None:
        alexn.load_state_dict(state_dict)
    return alexn
