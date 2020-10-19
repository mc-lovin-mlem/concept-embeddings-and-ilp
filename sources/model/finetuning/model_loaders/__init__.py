"""Loaders for models modified for fine-tuning on the picasso dataset.
Models are such trained on ImageNet-like datasets (input: 224x224 images).
"""
from typing import NamedTuple, Callable, Sequence, Tuple, Dict

import torch

from .alexnet import modified_alexnet, ALEXNET_FINETUNE_LAYERS
from .mobilenetv2 import modified_mobilenetv2, MOBILENETV2_FINETUNE_LAYERS
from .resnext import modified_resnext50, RESNEXT_FINETUNE_LAYERS
from .vgg16 import modified_vgg16, VGG16_FINETUNE_LAYERS
from .vggface import modified_vggface, VGGFACE_FINETUNE_LAYERS


class ModelInfo(NamedTuple):
    """Default information needed to acquire, fine-tune, and analyze a model on picasso dataset."""
    loader: Callable[..., torch.nn.Module]
    """The function to load an instance of the model modified for the picasso dataset."""
    finetune_layers: Sequence[str]
    """The default layer IDs to fine-tune (must be keys in `dict(main_model.named_modules())`)."""
    analysis_layers: Tuple[str, ...]
    """The IDs of layers that may be of interest for concept analysis
    (must be keys in `dict(main_model.named_modules())`)."""


MODELS_INFO: Dict[str, ModelInfo] = dict(
    VGG16=ModelInfo(
        loader=modified_vgg16,
        finetune_layers=VGG16_FINETUNE_LAYERS,
        analysis_layers=(
            'features.11',  # =conv3_1 after ReLU
            'features.13',  # =conv3_2 after ReLU
            'features.15',  # =conv3_3 after ReLU
            'features.18',  # =conv4_1 after ReLU
            'features.20',  # =conv4_2 after ReLU
            'features.22',  # =conv4_3 after ReLU
            'features.25',  # =conv5_1 after ReLU
            'features.27',  # =conv5_2 after ReLU
            'features.29',  # =conv5_3 after ReLU
        ),
    ),
    VGGFace=ModelInfo(
        loader=modified_vggface,
        finetune_layers=VGGFACE_FINETUNE_LAYERS,
        analysis_layers=(
            'conv3.convs.0',
            'conv3.convs.1',
            'conv3.convs.2',
            'conv4.convs.0',
            'conv4.convs.1',
            'conv4.convs.2',
            'conv5.convs.0',
            'conv5.convs.1',
            'conv5.convs.2',
        ),
    ),
    AlexNet=ModelInfo(
        loader=modified_alexnet,
        finetune_layers=ALEXNET_FINETUNE_LAYERS,
        analysis_layers=(
            'features.4',  # =conv 2 after ReLU
            'features.7',  # =conv 3 after ReLU
            'features.9',  # =conv 4 after ReLU
        ),
    ),
    ResNeXt=ModelInfo(
        loader=modified_resnext50,
        finetune_layers=RESNEXT_FINETUNE_LAYERS,
        analysis_layers=(
            'layer2.0',
            'layer2.1',
            'layer2.2',
            'layer2.3',
            'layer3.0',
            'layer3.1',
            'layer3.2',
            'layer3.3',
            'layer3.4',
            'layer3.5',
            'layer4.0',
            'layer4.1',
            'layer4.2',
        )
    ),
    MobileNetV2=ModelInfo(
        loader=modified_mobilenetv2,
        finetune_layers=MOBILENETV2_FINETUNE_LAYERS,
        analysis_layers=(
            'features.4',
            'features.5',
            'features.6',
            'features.7',
            'features.8',
            'features.9',
            'features.10',
            'features.11',
            'features.12',
            'features.13',
            'features.14',
            'features.15',
            'features.16',
            'features.17',
            'features.18',
        )
    )
)
"""Collection of model information for all available models."""
