"""Default settings common to all models."""

HIDDEN_DIM = 512
"""Default number of hidden units as in- and output to the 2nd VGG(Face) dense layer.
The original value is 4096."""
NUM_CLASSES = 1
"""Constant: number of output classes; set to 1 for binary classification with one output node 
(0=negative class, 1=positive class)."""

DEVICE: str = 'cpu'
"""Default device to use for training and evaluation."""
MODEL_PKL_TEMPL: str = "{model_lower}_finetuned.pkl"
"""The template to find finetuned .pkl model files for a given model class."""

EPOCHS = 1
"""Default number of epochs for fine-tuning."""
BATCH_SIZE = 32
"""Default batch size for fine-tuning on picasso dataset."""
