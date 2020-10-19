"""Scripts and modules for finetuning a VGG face model for the picasso dataset."""

from . import defaults
from .finetune_pytorch import finetune, picasso_data_loader, \
    evaluate, train_one_epoch, set_trainable
