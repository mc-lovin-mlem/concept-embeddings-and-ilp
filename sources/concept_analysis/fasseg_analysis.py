"""Wrapper and default settings for concept analysis on picasso dataset."""
# external packages
import logging
import os
import sys
from datetime import datetime
from typing import Tuple, Optional, Any, Dict, Union

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

plt.rcParams['figure.dpi'] = 300
to_tens = torchvision.transforms.ToTensor()

PROJECT_ROOT = "."  # assume script is called from within project root
"""The project root directory path relative to the folder from which the script is called."""
# for imports from concept embeddings sub-project:
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sources", "concept_analysis", "ext_lib"))

# project internal imports:
from ..dataset import PicassoMaskHandle

# imports from concept embeddings sub-project:
from hybrid_learning.concepts.analysis import ConceptAnalysis
from hybrid_learning.concepts import models
from hybrid_learning.concepts.concepts import SegmentationConcept2D as Concept
from hybrid_learning.concepts.kpis import BalancedBCELoss, WeightedLossSum, TverskyLoss
from hybrid_learning.concepts.embeddings import ConceptEmbedding
from hybrid_learning.datasets import transforms, DatasetSplit, \
    DataTriple


# HELPER FUNCTIONS
# ----------------

def train_val_args_gen(concept_rel_size: Tuple[float, float], factor_pos_class: float,
                       learning_rate: float, intersect_encode_post_thresh: float,
                       input_image_size: Tuple[float, float], weight_decay: float,
                       loss_weights: Dict[str, float], max_epochs: int,
                       factor_false_positives: float = None,
                       device: Union[torch.device, str] = 'cpu',
                       **unused_args):
    """Obtain the arguments for the concept handle specific to the used concept."""
    # derived value, do not change:
    factor_false_positives = factor_false_positives if factor_false_positives is not None \
        else factor_pos_class
    _kernel_size: Tuple[int, int] = (int(concept_rel_size[0] * input_image_size[0]),
                                     int(concept_rel_size[1] * input_image_size[1]))
    # Arguments to the concept model train test handle
    train_val_args = dict(
        max_epochs=max_epochs,
        device=device,
        loss_fn=WeightedLossSum([BalancedBCELoss(factor_pos_class=factor_pos_class),
                                 TverskyLoss(factor_false_positives=factor_false_positives)],
                                [loss_weights['BalancedBCELoss'], loss_weights['TverskyLoss']]),
        optim_handle=models.ResettableOptimizer(torch.optim.Adam, lr=learning_rate,
                                                weight_decay=weight_decay),
        # Transformation applied to the ground truth masks
        transforms=transforms.OnTarget(transforms.IntersectEncode(
            post_thresh=intersect_encode_post_thresh,
            normalize_by='target', kernel_size=_kernel_size)),
        # transforms = transforms.IoUEncode(post_thresh=0.4, kernel_size=_kernel_size),
        # Transformation applied to the tuples (model output, ground truth)
        model_output_transform=transforms.SameSize(resize_target=False, interpolation="bilinear"),
    )
    return train_val_args


def concept_gen(input_image_size: Tuple[int, int],
                part_name: str, fasseg_root: str,
                concept_rel_size: Optional[Tuple[float, float]],
                fasseg_handle: type, train_folder: str, test_folder: str,
                dataset_args: Dict[str, Any] = None,
                **unused_args):
    """Generate a FASSEG dataset concept for given part and settings."""

    # The main model will require a specific input size and format.
    # The following specifies the transformation to apply to FASSEG data such that
    # the images can be fed to the selected model.
    def fasseg_transforms(img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trafo for ImageNet-like data: Pad to square, resize to 224x224 and binarize mask."""
        img_t = transforms.PadAndResize(input_image_size)(to_tens(img))
        mask_t = transforms.PadAndResize(input_image_size, interpolation=Image.NEAREST)(to_tens(mask))
        return img_t, mask_t

    dataset_args = dataset_args if dataset_args is not None else {}
    dataset_args.update(dict(part_name=part_name, transforms=fasseg_transforms))
    concept_train_val_data = fasseg_handle(
        split=DatasetSplit.TRAIN_VAL, **dataset_args,
        dataset_root=os.path.join(fasseg_root, train_folder))
    concept_test_data = fasseg_handle(
        split=DatasetSplit.TEST, **dataset_args,
        dataset_root=os.path.join(fasseg_root, test_folder))

    concept = Concept(name=part_name,
                      data=DataTriple(train_val=concept_train_val_data,
                                      test=concept_test_data),
                      rel_size=concept_rel_size)
    return concept


DEFAULT_SETTINGS = dict(
    parts={
        # The part to work with
        'NOSE': dict(
            # The usual concept size relative to the image size in (height, width)
            concept_rel_size=(0.05, 0.15),
            # Ratio of background to total pixels in transformed ground truth masks
            factor_pos_class=0.999,  # TODO: automatically calculate
            # Threshold applied to ground truth masks after intersection encoding
            intersect_encode_post_thresh=0.5
        ),
        'MOUTH': dict(
            concept_rel_size=(0.08, 0.2),
            factor_pos_class=0.995,
            intersect_encode_post_thresh=0.8
        ),
        'EYES': dict(
            concept_rel_size=(0.08, 0.16),
            factor_pos_class=0.995,
            intersect_encode_post_thresh=0.35  # half, since there are 2 eyes in the image!
        )
    },
    # The maximum number of epochs to run
    max_epochs=2,
    # The learning rate to apply for the analysis;
    # for concept specific settings put into parts settings
    learning_rate=0.001,
    # The weight decay to apply (L1 regularization)
    weight_decay=0.,
    # If given, for the Tversky loss the factor for the false positive rate in [0,1];
    # if >0.5, weights false positives more than false negatives, if <0.5 the other way round.
    factor_false_positives=0.5,
    # The weights for the summands of the loss; keys should match the loss class names
    loss_weights={"BalancedBCELoss": 0.1, "TverskyLoss": 1.},

    # # DATASET SETTINGS (FASSEG training images)
    # # The handle class to use for the segmentation data
    # fasseg_handle=fasseg.FASSEGHandle,
    # # The root directory of the train and test data folders
    # fasseg_root=sys.path.insert(0, os.path.join(PROJECT_ROOT, "sources", "concept_analysis", "ext_lib",
    #                                             "dataset", "FASSEG", "V1"))
    # # The train and test folders.
    # # Swap for FASSEG, since the original paper featured only 20 training but 50 test samples.
    # test_folder="Train_RGB",
    # train_folder="Test_RGB",
    # DATASET SETTINGS (Masked Picasso training images)
    # The handle class to use for the segmentation data
    fasseg_handle=PicassoMaskHandle,
    # The root directory of the train and test data folders
    fasseg_root=os.path.join(PROJECT_ROOT, "dataset", "picasso_dataset_wt_masks"),
    # The train and test folders.
    test_folder="test",
    train_folder="train",
    # Any further arguments common to both train/val and test data set.
    # E.g. for PicassoMaskHandle, specify the class to take images from (e.g. 'neg'=negative only)
    dataset_args={'restrict_cls_to': 'all'},

    # The size input images to the model must have in (width, height) in px
    input_image_size=(224, 224),

    # TRAINING SETTINGS
    # How many concept models to train for each layer before taking the mean:
    # The number of cross-validation runs to conduct (runs with distinct validation set)
    cross_val_runs=3,
    # The number of validation data splits per cross-validation run
    # (validation data proportion then is 1/num_val_splits)
    num_val_splits=5,
    # Whether to show training progress bars during analysis (not logged)
    show_train_progress_bars=True,
    # If a part does not specify its concept or train_val_args, use *_gen to generate it
    # Will be called as concept_gen(part_name=part_name, **{**DEFAULT_SETTINGS, **parts[part_name]})
    concept_gen=concept_gen,
    train_val_args_gen=train_val_args_gen
)
"""The default settings to use for analysis.
Specifies settings format needed for :py:func:`analysis` func.
To conduct a run of :py:func:`analysis`, ``main_model`` and ``layer_infos`` 
must be specified in addition. 
"""


def analysis(setts: Dict[str, Any],
             logger: logging.Logger = None,
             logging_formatter: logging.Formatter = None,
             exp_root_templ: str = os.path.join("experiments", "concept_analysis",
                                                "fasseg_analysis_{time}"),
             common_log_file: str = 'log.txt'
             ) -> Tuple[str, Dict[str, ConceptEmbedding]]:
    """Conduct an analysis according to given settings, log and save results.
    For details on the outputs of the single concept runs see
    ConceptAnalysis.best_embedding_with_logging().

    :param setts: settings; must contain the keys provided by module level DEFAULT_SETTINGS
        and additionally the 'main_model' and a list with 'layer_infos'
        (may be a list with string layer IDs).
    :param logger: the logger to use for file logging;
        for the analysis, the logging level is set to logging.INFO
    :param common_log_file: the log file to use to collect all analysis logging activities
    :param logging_formatter: formatter for the logging file handles for file logging;
        defaults to the formatter of the first existing handle of the module level logger
    :param exp_root_templ: the string formatting template to create the experiment root folder;
        must contain '{time}' into which a time string will be inserted
    """
    # region Value init and check
    # Are all specified model layers valid?
    main_model_modules = list(dict(setts['main_model'].named_modules()).keys())
    for layer_id in setts['layer_infos']:
        assert layer_id in [name for name in main_model_modules], \
            "Layer_id {} not in model".format(layer_id)
    exp_root: str = exp_root_templ.format(time=datetime.now().strftime("%Y-%m-%d_%H%M%S%f"))
    os.makedirs(exp_root)  # Throw error if exists
    # endregion

    # region Logging settings
    logger = logger or logging.getLogger(__name__)
    orig_logging_level: int = logger.level
    logger.setLevel(logging.INFO)
    common_log_file_handler = logging.FileHandler(os.path.join(exp_root, common_log_file))
    common_log_file_handler.setLevel(logging.INFO)
    if len(logger.handlers) > 0 and logging_formatter is None:
        logging_formatter = logger.handlers[0].formatter
    common_log_file_handler.setFormatter(logging_formatter)
    logger.addHandler(common_log_file_handler)
    # endregion

    best_embeddings: Dict[str, ConceptEmbedding] = {}
    for concept_name, concept_info in setts['parts'].items():
        concept_exp_root = os.path.join(exp_root, concept_name)
        curr_settings: Dict[str, Any] = dict(part_name=concept_name, **{**setts, **concept_info})
        concept = concept_info.get('concept', concept_gen(**curr_settings))
        train_val_args = concept_info.get('train_val_args', train_val_args_gen(**curr_settings))

        analyzer = ConceptAnalysis(
            concept=concept, model=setts['main_model'],
            layer_infos=setts.get('layer_infos', None),
            cross_val_runs=setts['cross_val_runs'], num_val_splits=setts['num_val_splits'],
            show_train_progress_bars=setts['show_train_progress_bars'],
            train_val_args=train_val_args
        )
        best_embeddings[concept_name] = analyzer.best_embedding_with_logging(
            concept_exp_root=concept_exp_root,
            logger=logger, file_logging_formatter=logging_formatter)

    logger.removeHandler(common_log_file_handler)
    logger.setLevel(orig_logging_level)
    return exp_root, best_embeddings
