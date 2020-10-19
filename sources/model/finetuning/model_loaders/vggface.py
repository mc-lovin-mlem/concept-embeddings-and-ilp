"""Loader of modified VGGFace for finetuning on picasso dataset."""
from typing import Sequence, Dict

import torch

from sources.model.vggface_pytorch.vggface import VggFace, vggface
from ..defaults import HIDDEN_DIM, NUM_CLASSES

VGGFACE_FINETUNE_LAYERS: Sequence[str] = ('fc1', 'fc2', 'fc3', 'conv5.convs.1', 'conv5.convs.2')
"""Layers to finetune (layer names from model.named_modules()) for modified VGG Face."""


def modified_vggface(state_dict: Dict[str, torch.Tensor] = None, *,
                     pretrained: bool = False,
                     num_classes: int = NUM_CLASSES, hidden_dim: int = HIDDEN_DIM) -> VggFace:
    """Modify a VGG face model to have num_classes output classes.
    A VggFace instance is created (initialized according to `pretrained`) and modified as follows:
    The last before final linear dense layers is exchanged for one with hidden_dim units in- and
    output, and the number of output classes is set to num_classes.

    :param pretrained: whether to initialize the model with the pretrained VggFace weights where
        applicable; overridden by state_dict
    :param state_dict: state dict with which to initialize parameters
    :param num_classes: number of output classes of the modified model (no sigmoid applied)
    :param hidden_dim: in- and output dimension of the second dense layer
    :return: the modified VggFace instance; all non-modified layers are initialized with the
        pretrained weights if pretrained is True
    """
    vggf: VggFace = vggface(pretrained=pretrained)
    # Add fine-tuning/transfer learning modules
    vggf.fc1 = torch.nn.Linear(7 * 7 * 512, hidden_dim)
    vggf.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
    vggf.fc3 = torch.nn.Linear(hidden_dim, num_classes)
    if state_dict is not None:
        vggf.load_state_dict(state_dict)
    return vggf
