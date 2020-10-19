"""Fine-tune all available models on picasso with default settings dataset if .pkl is missing.
Call as
`python script/finetune_all_models.py`
to fine-tune all known models with default values.
Resulting .pkl files are stored in MODELS_ROOT folder.
"""
import argparse
import os
import sys

PROJECT_ROOT = "."  # assume, file is called from within project root
MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')
"""Root where to search for .pkl files and store them in."""
PICASSO_ROOT = os.path.join(PROJECT_ROOT, "dataset", "picasso_dataset")
"""Root directory of the picasso dataset."""
sys.path.insert(0, PROJECT_ROOT)

# project internal imports:
from sources.model.finetuning import picasso_data_loader, finetune
from sources.model.finetuning import defaults
from sources.model.finetuning.model_loaders import MODELS_INFO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', default=list(MODELS_INFO.keys()),
                        choices=list(MODELS_INFO.keys()),
                        help='The models to create.')
    parser.add_argument('--models_root', type=str, default=MODELS_ROOT,
                        help='The root directory where to search and store model .pkl files.')
    parser.add_argument('--picasso_root', type=str, default=PICASSO_ROOT,
                        help='The root directory of the picasso dataset for fine-tuning.')
    parser.add_argument('--device', type=str, default=defaults.DEVICE,
                        help='The device to run fine-tuning on.')
    parser.add_argument('--model_pkl_templ', type=str, default=defaults.MODEL_PKL_TEMPL,
                        help='The string template for the file name (or path relative to '
                             'models_root) of a model .pkl file. Must contain "{model_lower}", '
                             'which is replaced by the lower-case model name.')
    args = parser.parse_args()

    # Data loaders
    loaders224 = {'train': picasso_data_loader('train', picasso_root=args.picasso_root),
                  'val': picasso_data_loader('test', picasso_root=args.picasso_root)}
    os.makedirs(args.models_root, exist_ok=True)

    # Binary classification on 1 output node
    for model_name in args.models:
        info = MODELS_INFO[model_name]
        pkl_file = os.path.join(args.models_root,
                                args.model_pkl_templ.format(model_lower=model_name.lower()))
        if os.path.exists(pkl_file):
            print("Found fine-tuned {} at {}, continuing.".format(model_name, pkl_file))
        else:
            print("Finetuning model {} (pkl file: {}) ...".format(model_name, pkl_file))
            finetune(model=info.loader(pretrained=True),
                     finetune_layers=info.finetune_layers, pkl_file=pkl_file,
                     loaders=loaders224, device=args.device)
