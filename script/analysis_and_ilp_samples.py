#!/usr/local/env python3
"""Script for concept analysis on picasso dataset with analysis pre- and post-processing.

The steps conducted are:
1. If the model is not yet finetuned, do fine-tuning and save model file.
2. If the samples to collect ILP background knowledge for are not yet selected, select them
  and copy them to a model-specific dataset folder.
3. Conduct a new analysis run and save results to an experiment folder.
4. Evaluate the resulting concept embeddings on the selected samples and store results in the
  experiment folder.

ILP samples and results for the samples are stored each in a folder named
`{model_lower}{pkl_hash}_ilp_samples`
where model_lower is the model identifier in lower case, and
pkl_hash are the first 8 letters of the hex md5 sum of the model's .pkl file.
"""
import argparse
import json
import logging
import os
import sys
from typing import Tuple, Dict

import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss

PROJECT_ROOT = "."
sys.path.insert(0, os.path.join(PROJECT_ROOT))  # FIXME: add to sources.__init__?
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sources", "concept_analysis", "ext_lib"))
from sources.model.finetuning.model_loaders import MODELS_INFO
from sources.model import finetuning
from sources.concept_analysis import ilp_samples
import sources.concept_analysis.fasseg_analysis as analysis
from hybrid_learning.concepts.models import ConceptDetectionModel2D as ConceptModel

LOGGING_FORMAT: str = '---------- \t{asctime}\n{levelname}: {message}'
"""The logging format in .format() style."""

parser = argparse.ArgumentParser(description='Collect concept predictions for selected samples.')
parser.add_argument('--model', type=str,
                    choices=list(MODELS_INFO.keys()), required=True,
                    help='The identifier of the model to conduct an analysis for.')
parser.add_argument('--device', type=torch.device,
                    default=torch.device('cpu'),
                    help='The device to run experiments on.')
parser.add_argument('--force_finetune', action='store_true')
parser.add_argument('--finetune_epochs', type=int,
                    default=finetuning.defaults.EPOCHS,
                    help='Number of epochs for fine-tuning if applied.')
parser.add_argument('--finetune_batch_size', type=int,
                    default=finetuning.defaults.BATCH_SIZE,
                    help='Batch size for fine-tuning if applied.')
parser.add_argument('--picasso_dataset_root', type=str,
                    default=os.path.join(PROJECT_ROOT, "dataset", "picasso_dataset"),
                    help='Path to the root directory of the picasso dataset for fine-tuning and '
                         'ILP sample selection.')
parser.add_argument('--experiment_root', type=str,
                    default=os.path.join(PROJECT_ROOT, "experiments", "concept_analysis"),
                    help='Path to the root directory for concept analysis experiments.')
parser.add_argument('--finetune_layers', type=str, nargs='+',
                    help='The IDs of the layers to fine-tune from model.named_modules()')
parser.add_argument('--num_ilp_samples_per_cls', type=int, default=50,
                    help='The number of ILP samples to choose per prediction class '
                         '(positive/negative predicted).')
parser.add_argument('--dataset_root', type=str,
                    default=os.path.join(PROJECT_ROOT, "dataset", "picasso_ilp_samples"),
                    help='Where to store and search for the folder with selected ILP samples.')
parser.add_argument('--model_root', type=str,
                    default=os.path.join(PROJECT_ROOT, "models"),
                    help='Where to store and search for model .pkl files and to put intermediate '
                         '.csv with model predictions.')
parser.add_argument('--analysis_settings', type=json.loads, default={},
                    help='Any simple settings for the analysis to override; '
                         'provide as dict in json format')


def prepare_logging(logging_format: str = LOGGING_FORMAT
                    ) -> Tuple[logging.Logger, logging.Formatter]:
    """Collect a logging formatter and a logger with a stream handle and correct logging level.
    Also set pandas options to logging-friedly values."""
    # logging settings
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_stream_handler = logging.StreamHandler(stream=sys.stdout)
    log_stream_handler.setLevel(logging.INFO)
    log_format = logging.Formatter(logging_format, style='{')
    log_stream_handler.setFormatter(log_format)
    logger.addHandler(log_stream_handler)
    return logger, log_format


if __name__ == "__main__":
    # SETTINGS:
    # ---------
    args: argparse.Namespace = parser.parse_args()
    model_defaults = MODELS_INFO[args.model]
    args.finetune_layers = args.finetune_layers or model_defaults.finetune_layers
    args.analysis_settings['layer_infos'] = \
        args.analysis_settings.get('layer_infos',
                                   model_defaults.analysis_layers)
    model_pkl_file = os.path.join(
        args.model_root, finetuning.defaults.MODEL_PKL_TEMPL.format(model_lower=args.model.lower()))
    # Logging settings:
    logger, log_format = prepare_logging()
    # pandas settings:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    logger.info("Starting script with settings:\n%s\n"
                "analysis_settings \t%s\n"
                "finetune_layers \t%s",
                pd.Series(vars(args)).drop(['analysis_settings', 'finetune_layers']).to_string(),
                args.analysis_settings, args.finetune_layers)

    # OBTAIN FINE-TUNED MODEL:
    # ------------------------
    logger.info("Collecting model (.pkl file: %s) ...", model_pkl_file)
    if os.path.exists(model_pkl_file) and not args.force_finetune:
        model = model_defaults.loader(state_dict=torch.load(model_pkl_file,
                                                            map_location=args.device))
    else:
        logger.info("Starting model fine-tuning ...")
        model = model_defaults.loader(pretrained=True)
        loader_args = dict(batch_size=args.finetune_batch_size,
                           picasso_root=args.picasso_dataset_root)
        loaders224 = {'train': finetuning.picasso_data_loader(split='train', **loader_args),
                      'val': finetuning.picasso_data_loader(split='test', **loader_args)}
        model = finetuning.finetune(
            model, device=args.device,
            finetune_layers=args.finetune_layers,
            pkl_file=model_pkl_file, loaders=loaders224,
            loss=BCEWithLogitsLoss(), epochs=args.finetune_epochs,
        )

    # GATHER MODEL IDENTIFIER:
    # ------------------------
    logger.info("Gathering model .pkl identifier ...")
    model_prefix = model.model_id(model_name=args.model, model_pkl_file=model_pkl_file)
    ilp_samples_dir = "{}_ilp_samples".format(model_prefix)

    # CONDUCT CONCEPT ANALYSIS:
    # -------------------------
    logger.info("Starting concept analysis ...")
    exp_root, best_embeddings = analysis.analysis(
        setts={**analysis.DEFAULT_SETTINGS,
               **args.analysis_settings,
               'device': args.device,
               'main_model': model},
        logging_formatter=log_format, logger=logger,
        exp_root_templ=os.path.join(args.experiment_root, model_prefix + "_{time}")
    )
    logger.info("Experiment root folder: %s", exp_root)

    # PICK SAMPLES FOR ILP:
    # ---------------------
    logger.info("Checking for existing selection of ILP samples ...")
    ilp_samples_root = os.path.join(args.dataset_root, ilp_samples_dir)
    if not os.path.exists(ilp_samples_root):
        logger.info("Selecting ILP samples and creating samples root (%s)...", ilp_samples_root)
        ilp_samples.create_samples_folder(
            model=model, device=args.device,
            dataset_root=args.picasso_dataset_root, dest_root=ilp_samples_root, splits=('test',),
            num_imgs_per_cls=args.num_ilp_samples_per_cls,
            csv_file=os.path.join(args.model_root, "{}_preds_test.csv".format(model_prefix)),
        )

    # GENERATE MASKS for the best embeddings and the given samples:
    # --------------
    logger.info("Starting generation of ILP samples ...")
    ilp_samples_results_root = os.path.join("experiments", "ilp_samples", os.path.basename(exp_root))
    best_c_models: Dict[str, ConceptModel] = {c: ConceptModel.from_embedding(emb)
                                              for c, emb in best_embeddings.items()}
    # Iterate over the ground truth classes (can be any combination of 'pos', 'neg')
    ilp_samples.generate(best_c_models,
                         # select ILP train samples from test set:
                         sample_root=os.path.join(ilp_samples_root, 'test'),
                         dest_root=os.path.join(ilp_samples_results_root))

    logger.info("DONE.")
