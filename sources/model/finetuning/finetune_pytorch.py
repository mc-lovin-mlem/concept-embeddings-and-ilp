"""Fine-tuning for models trained on ImageNet on the picasso dataset of the project."""
import os
from typing import Sequence, Tuple, Dict, Union

import numpy as np
import torch
from keras.preprocessing.image import ImageDataGenerator
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm

# Default settings:
from .defaults import EPOCHS, BATCH_SIZE


def to_pytorch_format(inputs: np.ndarray, targets: np.ndarray
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn tensorflow encoded input images and tensor targets into pytorch encoding."""
    # transform data to pytorch tensors
    targets = torch.from_numpy(targets).float()
    # torch tensors are transposed compared to keras -- in all but the batch dim
    inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).contiguous().float()
    return inputs, targets


def evaluate(model: torch.nn.Module, data_loader: Sequence,
             tf_data_format: bool = False):
    """Collect the accuracy value of the model on a given one-hot-encoded classification dataset.
    WARNING: If the model does binary classification with one output node, the output is expected
    to be that of a sigmoid.
    Operations are done on the device of the first of the model's parameters.

    :param model: the model to evaluate
    :param data_loader: the data loader sequence; must support __getitem__ and yield
        tuples of input and target batch tensors, where inputs are 3-channel 2D images
        and targets are one-hot-encoded vectors.
    :param tf_data_format: whether to assume tensorflow format of the tensors,
        i.e. numpy tensors, channels last, and (width, height) instead of (height, width)
    :return: accuracy over the complete dataset
    """
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cpu'
    model.eval()

    tp, tn, all_ = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm((data_loader[i] for i in range(len(data_loader))),
                                    total=len(data_loader)):
            if tf_data_format:
                inputs, targets = to_pytorch_format(inputs, targets)
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            # actual prediction
            preds = model(inputs)

            # Calc true positives, true negatives, and tn+tp+fn+fp for batch
            _, batch_tp, batch_tn, batch_all = accuracy(preds, targets)
            # add to accumulated values:
            tp += batch_tp
            tn += batch_tn
            all_ += batch_all
    # accuracy:
    acc = float(tp + tn) / all_
    return acc


def train_one_epoch(model: torch.nn.Module, data_loader: Sequence,
                    optimizer, loss: torch.nn.Module,
                    tf_data_format: bool = False, keep_one_hot: bool = False):
    """Train the model on a given one-hot-encoded classification dataset.

    :param loss: loss function to use
    :param optimizer: optimizer instance to use
    :param model: the model to evaluate
    :param data_loader: the data loader sequence; must support __getitem__ and yield
        tuples of input and target batch tensors, where inputs are 3-channel 2D images
        and targets are one-hot-encoded vectors.
    :param tf_data_format: whether to assume tensorflow format of the tensors,
        i.e. numpy tensors, channels last, and (width, height) instead of (height, width)
    :param keep_one_hot: the dataset targets are supposed to be one-hot encoded; whether to keep
        them one-hot encoded or encode the class in a 1D tensor via argmax();
        for torch.nn.CrossEntropy() set to False
    :return: accuracy over the complete dataset
    """
    if len(list(model.parameters())) == 0:
        raise ValueError("Model to train does not have any parameters!")
    device = next(model.parameters()).device
    model.train()

    pbar: tqdm = tqdm((data_loader[i] for i in range(len(data_loader))), total=len(data_loader))
    for inputs, targets in pbar:

        if tf_data_format:
            inputs, targets = to_pytorch_format(inputs, targets)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # calc loss and update model
        optimizer.zero_grad()
        preds = model(inputs)
        loss_tensor = loss(preds if preds.size()[1] > 1 else preds.squeeze(1),
                           targets if keep_one_hot else torch.argmax(targets, 1).float())
        loss_tensor.backward()
        optimizer.step()

        # log stats:
        batch_acc, _, _, _ = accuracy(preds, targets)
        pbar.set_postfix({'loss': loss_tensor.item(), 'acc': batch_acc})
        pbar.update()


def accuracy(preds: torch.Tensor, targets: torch.Tensor, cls_dim: int = 1,
             ) -> Tuple[float, float, float, float]:
    """Calculate accuracy for given predictions and one-hot-encoded binary targets.
    The class dimension is assumed to be in dimension cls_dim.
    If the size of cls_dim is 1 in the predictions, it is assumed to be a binary classification
    output in one node (non-one-hot-encoded) and the values are sigmoided and thresholded by 0.5 to
    obtain the class prediction.

    :param preds: predictions tensor
    :param targets: targets tensor
    :param cls_dim: the dimension in which to find the class information in case of multi-class
        classification; e.g. if cls_dim == 1 the prediction tensors are assumed to be of shape
        (batch, num_classes, *)
    :return: tuple of (accuracy, #true pos, #true neg, total number of samples)
    """
    # Calc true positives, true negatives, and tn+tp+fn+fp for batch
    # from one-hot or binary encoding to class indices:
    class_pred = torch.argmax(preds, dim=cls_dim) if preds.size()[cls_dim] > 1 else \
        (torch.sigmoid(preds) > 0.5).squeeze(cls_dim).float()

    class_gt = torch.argmax(targets, dim=cls_dim).float()
    # accuracy calculation
    batch_tp = float(torch.sum(class_pred * class_gt))
    batch_tn = float(torch.sum((1 - class_pred) * (1 - class_gt)))
    batch_all = float(class_gt.size()[0])
    batch_acc = (batch_tp + batch_tn) / batch_all
    return batch_acc, batch_tp, batch_tn, batch_all


def picasso_data_loader(split: str, picasso_root: str, batch_size: int = BATCH_SIZE,
                        target_size: Tuple[int, int] = (224, 224)) -> Sequence:
    """Get a data loader for the picasso dataset (in tensorflow format).
    The loader yields batches of tuples of input image of size 224x224 and the
    output class (one-hot-encoded).

    :param split: the dataset split to use, either 'train' or 'test'
    :param batch_size: the batch size to apply
    :param picasso_root: the path to the root directory of the picasso dataset to load
    :param target_size: the size of the images to load
    """
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        os.path.join(picasso_root, split),
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')
    return validation_generator


def set_trainable(model: torch.nn.Module, layer_or_param_names: Sequence[str]):
    """Set only the layers or parameters named in layer_or_parameter_names to trainable.
    All others are set to non-trainable (i.e. requires_grad=False).

    :param layer_or_param_names: Sequence with the names of parameters as keys in
        model.named_parameters() or names of complete sub-modules as keys in
        model.named_modules().
    :param model: the model for which to set the parameters
    :raises: ValueError if one of the names is invalid (no module or parameter name)
    """
    # value check
    for name in layer_or_param_names:
        is_mod_name = (name in list(dict(model.named_modules()).keys()))
        is_param_name = (name in list(dict(model.named_parameters()).keys()))
        if not is_mod_name or is_param_name:
            raise ValueError("name {} does not occur in sub-module or parameter list of {}"
                             .format(name, str(model)))

    # Set all parameters to non-trainable:
    for p in model.parameters():
        p.requires_grad = False

    # Set the selected modules to trainable:
    for m_name, module in model.named_modules():
        if m_name in layer_or_param_names:
            for p in module.parameters():
                p.requires_grad = True
    # Set selected parameters to trainable:
    for p_name, param in model.named_parameters():
        if p_name in layer_or_param_names:
            param.requires_grad = True


def finetune(model: torch.nn.Module, finetune_layers: Sequence[str], loaders: Dict[str, Sequence],
             epochs: int = EPOCHS,
             loss: torch.nn.Module = BCEWithLogitsLoss(),
             keep_one_hot: bool = False,
             pkl_file: str = None,
             device: Union[torch.device, str] = torch.device('cpu')) -> torch.nn.Module:
    """Fine-tune all finetune_layers of model on the given one-hot-encoded classification dataset.
    If pkl_file is given: After each epoch, the model's state_dict is checkpointed to pkl_file if
    it has better accuracy than in the previous epochs.

    :param epochs: number of epochs to train
    :param loaders: dict of dataset loaders for keys 'train' and 'val';
        loaders must yield one-hot-encoded classification data
    :param pkl_file: file for model state checkpointing
    :param model: the model to finetune; loss must accept model output
    :param finetune_layers: names of layers (sub-modules) or parameters as in
        model.named_modules() respectively model.named_parameters()
    :param loss: loss for model outputs and targets; must either accept one-hot-encoded
        classification targets or keep_one_hot must be False (e.g. for CrossEntropyLoss())
    :param keep_one_hot: whether the given loss requires one-hot-encoded targets or not
    :param device: the device to run training and evaluation on
    :return: the fine-tuned model; if checkpointing is enabled (pkl_file not None), the best
        checkpoint is loaded and returned
    """
    # Trainable settings
    model.to(device)  # must be done before assigning parameters to optimizer!
    set_trainable(model, finetune_layers)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params)

    # Finetune on picasso dataset:
    best_val_acc = -1
    for epoch in range(epochs):
        train_one_epoch(model, loaders['train'], tf_data_format=True,
                        optimizer=optimizer,
                        loss=loss, keep_one_hot=keep_one_hot)

        # Evaluate and log:
        val_acc = evaluate(model, loaders['val'], tf_data_format=True)
        print("Epoch {}: val_acc={}".format(epoch, val_acc))

        # Possibly checkpoint:
        if pkl_file is not None and best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), pkl_file)
    # Load best checkpoint:
    if pkl_file is not None:
        model.load_state_dict(torch.load(pkl_file))
    return model
