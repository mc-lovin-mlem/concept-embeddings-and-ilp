"""This module collects functions to generate and load concept mask samples for ILP."""
import os
import sys
from typing import List, Dict, Any, Union

import PIL.Image
import numpy as np
import pandas as pd
import torch
import torchvision as tv
from PIL import Image
from tqdm import tqdm

# EVIL HACK to be able to include hybrid learning package without installation;
# does not work if working directory is not the PROJECT_ROOT!!!
PROJECT_ROOT = "."  # assert script is called from project root
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sources", "concept_analysis", "ext_lib"))
from hybrid_learning.concepts.models import ConceptDetectionModel2D as ConceptModel

to_tens = tv.transforms.ToTensor()


def to_img(tens: torch.Tensor) -> Image.Image:
    """Transform a (cuda) tensor to a PIL image."""
    trafo = tv.transforms.ToPILImage()
    return trafo(tens.cpu())


def load(results_root: str, gt_class_dirs=('pos', 'neg')):
    """Load the results from a concept analysis experiment into a pd.DataFrame.

    Each row of the DataFrame corresponds to one sample.
    The DataFrame has columns:

    - img_fn: the path to the original image relative to the dataset_root
    - pred: the float prediction of the model in [0,1]
    - ground_truth: the ground truth of the sample
    - <part name>: the mask for that part as PIL.Image.Image
      (original size; requires resizing to be applied to image)

    The results_root is assumed to have the following folder structure
    results_root > ground_truth_class > .npz files

    :param results_root: the root folder in which to find folders for each ground truth class
    :param gt_class_dirs: the names of the ground truth class directories to scrape;
        the names are also used as the 'ground_truth' entry in the DataFrame.
    :returns: the DataFrame with the results.
    """
    results_dicts: List[Dict[str, Any]] = []

    for gt_class in gt_class_dirs:
        cls_dir = os.path.join(results_root, gt_class)
        res_fns = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]

        # Each .npz entry looks like this:
        for res_fn in tqdm(res_fns, desc='Loading class {}'.format(gt_class)):
            # The results from results_root:
            res = np.load(os.path.join(cls_dir, res_fn))
            pred = float(res['pred'])  # for new format remove the sigmoid
            masks = {}
            for r, v in ((r, v) for r, v in res.items() if r != 'pred'):
                threshed_mask_np = (res[r] / 255) > 0.5
                masks[r] = PIL.Image.fromarray(threshed_mask_np)

            results_dicts.append({
                'img_fp': os.path.join(gt_class, res_fn.replace('.npz', '.png')),
                # the path to the original image relative to dataset_root
                'pred': pred,  # the prediction of the model for that sample (in [0, 1])
                'ground_truth': gt_class,  # the ground truth class of that sample
                **masks  # one column for each part
            })

    results: pd.DataFrame = pd.DataFrame(results_dicts)
    return results


def generate(named_c_models: Dict[str, ConceptModel],
             sample_root: str, dest_root: str,
             overwrite: bool = False,
             device: Union[torch.device, str] = torch.device('cpu'),
             recursive: bool = True):
    """Eval all .png samples in sample_root on all embeddings and the main model and save results.

    The results for image sample_root/<img_name>.png are saved using numpy.save() as
    dest_root/<img_name>.npz with the following format.
    When loading the result file using numpy.load(), one obtains a dict-like with the following
    keys:

    - 'pred': the prediction of the main model with sigmoid applied as single float value between
      0 (negative class) and 1 (positive class); to binary threshold at 0.5
    - <concept_name>: for each concept name, the corresponding concept predicted mask is saved as
      numpy array with values between 0 (no concept) and 1 (concept);
      the mask can be turned into an image using PIL.Image.fromarray() and has corresponding format;
      the mask size may differ from the size of the image, so apply a resize before comparison;
      to binarize threshold at 0.5 (resp. upsample and then threshold at 0.5);

    :param named_c_models: dictionary of {concept name: concept model} for the concept models to
        use for evaluation; the keys are used as keys within the resulting .npz files;
        the main_model_stump of each concept model must be non-None
    :param sample_root: the root directory under which to find the sample .png files
    :param dest_root: the root directory under which to store the generated .npz files
    :param recursive: whether to recursively search for .png files in subfolders
    :param overwrite: whether to overwrite existing .npz files
    :param device: the device to use for model inferences, or the string device specifier
    """
    os.makedirs(dest_root, exist_ok=True)

    # Init models
    assert all([c.main_model is not None for c in named_c_models.values()])
    model: torch.nn.Module = list(named_c_models.values())[0].main_model
    assert all([c.main_model is model for c in named_c_models.values()])
    # Take care to choose the same device:
    model.eval().to(device)
    for c_model in named_c_models.values():
        c_model.eval().to(device)

    sub_roots: List[str] = [sample_root] if not recursive else \
        [d for d, _, _ in os.walk(sample_root, followlinks=True)]
    for sub_root in sub_roots:
        rel_sub_root: str = os.path.relpath(sub_root, sample_root)
        os.makedirs(os.path.join(dest_root, rel_sub_root), exist_ok=True)
        sample_fps: List[str] = [os.path.join(sub_root, fn) for fn in os.listdir(sub_root)
                                 if fn.endswith(".png")]
        for sample_fp in tqdm(sample_fps, desc="Samples ({})".format(rel_sub_root)):
            # get destination
            npz_fp = os.path.join(dest_root,
                                  os.path.relpath(os.path.dirname(sample_fp), sample_root),
                                  os.path.basename(sample_fp).replace('.png', '.npz'))
            if os.path.exists(npz_fp) and not overwrite:
                raise FileExistsError("Overwrite is disabled and file {} exists".format(npz_fp))

            # collect content
            sample_t = to_tens(Image.open(sample_fp)).to(device)
            preds = _generate_one_sample(model, named_c_models, sample_t)

            # save
            with open(npz_fp, 'w+b' if overwrite else 'xb') as f:
                np.savez(f, **preds)


def _generate_one_sample(main_model: torch.nn.Module,
                         named_c_models: Dict[str, torch.nn.Module],
                         sample_t: torch.Tensor) -> Dict[str, np.ndarray]:
    """Evaluate all given concept models and their main model on the sample image.
    The image must be given as torch.Tensor which lies on the same device as the models.
    The resulting format is:
    {
    'pred': main_model output as numpy array of a float value in [0,1]),
    **{key: named_c_models[key] output as numpy array of a single-channel mask (PIL.Image)}
    }
    """
    with torch.no_grad():
        preds: Dict[str, np.ndarray] = \
            {'pred': (torch.sigmoid(main_model.eval()(sample_t.unsqueeze(0)))
                      .squeeze(0).detach().cpu().numpy())}
        for c_name, c_model in named_c_models.items():
            act_map = c_model.main_model_stump.eval()(sample_t.unsqueeze(0))
            mask_t = c_model.eval()(act_map).squeeze(0)
            # Convert the torch tensor to a PIL.Image.fromarray() compatible format:
            # noinspection PyTypeChecker
            preds[c_name] = np.array(to_img(mask_t))
        return preds
