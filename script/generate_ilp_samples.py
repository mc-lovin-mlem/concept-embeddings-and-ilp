"""Script to evaluate concept embeddings from an analysis run on selected samples for ILP.
Call from project root directory.
"""
import argparse
import os
import sys
from typing import Dict

import pandas as pd

pd.set_option('display.max_colwidth', None)
import torch

PROJECT_ROOT = os.path.join(".")  # assume script is called from project root dir
sys.path.insert(0, os.path.join(PROJECT_ROOT))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sources", "concept_analysis", "ext_lib"))

from sources.model.finetuning.defaults import MODEL_PKL_TEMPL
from sources.model.finetuning import model_loaders
from sources.concept_analysis.ilp_samples import generate
from hybrid_learning.concepts.embeddings import ConceptEmbedding
from hybrid_learning.concepts.models import ConceptDetectionModel2D as ConceptModel

MODELS_ROOT = "models"
"""The default root directory relative to project root under which to search for .pkl files."""
EXPS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "concept_analysis")
"""The default root directory relative to project root under which to search for experiment 
folders."""

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True,
                    choices=list(model_loaders.MODELS_INFO.keys()),
                    help='The identifier of the model to use.')
parser.add_argument('--pkl_file', type=str,
                    help='Path to the .pkl file of the main model. If not given, '
                         'it is tried to be automatically inferred using the model specifier.')
parser.add_argument('-e', '--exp_root', type=str, required=True,
                    help='Path to the experiment folder under which to look for the embeddings; '
                         'provide in system specific path syntax')
parser.add_argument('-i', '--samples_root', type=str,
                    default=os.path.join(PROJECT_ROOT, 'dataset', 'picasso_dataset'),
                    help='The root directory under which to find the samples to process; '
                         'provide in system specific path syntax')
parser.add_argument('-o', '--dest_root', type=str,
                    help='The destination directory under which to store the results of sample '
                         'evaluation; defaults to exp_root/all_ilp_samples')
parser.add_argument('-r', '--recursive', action='store_true',
                    help='Whether to recursively proceed into sub-folders to find images. '
                         'The found hierarchy is mimicked in the destination folder.')
parser.add_argument('-f', '--force', action='store_true',
                    help='Whether existing output files may be overwritten.')
parser.add_argument('--device', type=torch.device, default=torch.device('cpu'),
                    help='The device to run inference on.')
parser.add_argument('--concepts', nargs='+', default=['NOSE', 'MOUTH', 'EYES'],
                    help='The concepts for which to evaluate the given samples.')

if __name__ == "__main__":
    args = parser.parse_args()
    args.dest_root = args.dest_root or os.path.join(args.exp_root, 'all_ilp_samples')
    pkl_file: str = (args.pkl_file or
                     os.path.join(PROJECT_ROOT, MODELS_ROOT,
                                  MODEL_PKL_TEMPL.format(model_lower=args.model.lower())))
    # region Value checks
    for concept in args.concepts:
        if concept not in os.listdir(args.exp_root):
            raise FileNotFoundError("Concept {} not found in exp_root {}"
                                    .format(concept, args.exp_root))
        if not os.path.isdir(os.path.join(args.exp_root, concept)):
            raise NotADirectoryError("Concept {} in exp_root {} is not a directory"
                                     .format(concept, args.exp_root))
    # endregion

    print("Starting sample generation with settings:")
    key_len = max([len(k) for k in vars(args)])
    print("\n".join([("{:" + str(key_len) + "}\t{}").format(k, v) for k, v in vars(args).items()]),
          flush=True)

    main_model: torch.nn.Module = model_loaders.MODELS_INFO[args.model].loader(
        torch.load(pkl_file, map_location=args.device))
    best_embeddings: Dict[str, ConceptEmbedding] = {
        concept: ConceptEmbedding.load(os.path.join(args.exp_root, concept, 'best.npz'))
        for concept in args.concepts}
    best_c_models: Dict[str, ConceptModel] = {
        concept: ConceptModel.from_embedding(emb, main_model=main_model)
        for concept, emb in best_embeddings.items()}

    generate(named_c_models=best_c_models,
             sample_root=args.samples_root, dest_root=args.dest_root,
             recursive=args.recursive, overwrite=args.force,
             device=args.device)
