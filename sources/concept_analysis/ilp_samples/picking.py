"""Functions to evaluate the main model and pick according samples for ILP.
Run as script from project root as
`python3 script/picking.py`

When called as script, will pick for each prediction class the same amount of samples from the test
set that are closest to the decision boundary and copy those images into a destination folder.
For the settings see the settings section.
"""
import os
import shutil
from typing import Union, List, Iterable

import PIL.Image
import numpy as np
import pandas as pd
import torch
import torchvision as tv
from tqdm import tqdm

import model
from sources.model.finetuning import model_loaders

to_tens = tv.transforms.ToTensor()
to_img = tv.transforms.ToPILImage()


def get_model_results(model: torch.nn.Module,
                      dataset_root: str,
                      save_as: str = None,
                      device: Union[torch.device, str] = None,
                      splits: Iterable[str] = ('train', 'test'),
                      gt_classes: Iterable[str] = ('pos', 'neg')) -> pd.DataFrame:
    """Collect predictions of main_model for all samples in dataset_root.
    The dataset_root is assumed to have the structure
    dataset_root > split > ground_truth_class > <image files>.

    :param model: the model to evaluate
    :param device: if given, the device to run on
    :param dataset_root: the root directory of the dataset
    :param save_as: optional .csv file path to save the results in (overwrites!)
    :param splits: the dataset splits to evaluate
    :param gt_classes: the ground truth classes to evaluate
    """
    sub_results: List[pd.DataFrame] = []
    with torch.no_grad():
        for split in splits:
            for gt_class in gt_classes:
                folder = os.path.join(dataset_root, split, gt_class)
                folder_res = get_model_results_for_folder(
                    model, folder, device=device, pbar_desc="{}, {}".format(split, gt_class))
                sub_results.append(folder_res.assign(split=split, ground_truth=gt_class))
    results = pd.concat(sub_results, ignore_index=True)
    if save_as is not None:
        results.to_csv(save_as)
    return results


def get_model_results_for_folder(model: torch.nn.Module, folder: str,
                                 device: Union[torch.device, str] = None,
                                 pbar_desc: str = None) -> pd.DataFrame:
    """Collect model float prediction for all image files in folder.
    The model must return a 2D tensor of size (batch_size, binary predictions).
    All non-directory files ending with '.png' in folder are assumed to be valid image files
    loadable by PIL.Image.open.

    :param model: the model to use
    :param device: if given, the device to move the model onto before evaluation
    :param folder: the folder to search for image files in
    :param pbar_desc: description for the progress bar
    :return: pd.DataFrame with columns 'img' (the file name of the image relative to the folder),
        and 'pred' (the float sigmoid of the prediction of the model).
    """
    with torch.no_grad():
        model.eval()
        if device is not None:
            model.to(device)
        img_fns = [fn for fn in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, fn)) and fn.endswith('.png')]
        row_list = []
        for img_fn in tqdm(img_fns, desc=pbar_desc):  # TODO: batch-processing
            img = PIL.Image.open(os.path.join(folder, img_fn))
            img_t = to_tens(img).to(device)
            pred_t = torch.sigmoid(model(img_t.unsqueeze(0)).squeeze(0))
            row_list.append({'img': img_fn, 'pred': float(pred_t)})
        return pd.DataFrame(row_list)


def select_by_decision_boundary(preds: pd.DataFrame, num_imgs: int) -> List[str]:
    """Return a list of image paths that are closest to model decision boundary.
    The paths are relative to the dataset root assumed in the prediction information.
    """
    preds = preds.assign(dist_to_border=lambda r: np.abs(r.pred - 0.5))
    # preds.nsmallest did weird things
    smallest = preds.sort_values(by=['dist_to_border']).head(num_imgs)
    # get relative paths:
    imgs = smallest.apply(lambda row: os.path.join(row.split, row.ground_truth, row.img), axis=1)
    return list(imgs)


def create_samples_folder(model: torch.nn.Module, dataset_root: str, dest_root: str,
                          num_imgs_per_cls: int,
                          splits: Iterable[str] = None,
                          csv_file: str = None, device: Union[str, torch.device] = None):
    """Select samples closest to decision boundary from dataset_root and copy them to dest_root.
    The resulting collections for each respected split can be used as samples_root for generating
    ILP samples from analysis results.

    For each prediction class (positive predictions > 0.5, negative predictions < 0.5)
    at most num_imgs_per_cls are collected.
    The folder hierarchy in dataset_root must be:
    dataset_root > split > ('pos'|'neg') > image files ending with .png;
    the split is the dataset split, and 'pos' holds samples with positive ground truth,
    'neg' samples with negative ground truth.
    This hierarchy is mirrored for the destination root.

    :param model: the model for which the samples must be close to the decision boundary
    :param dataset_root: the root directory holding the samples (hierarchy described above)
    :param dest_root: the root directory to which to copy selected samples; must not exist!
    :param num_imgs_per_cls: the number of images predicted positive resp. negative to select
    :param splits: splits for which to select samples; defaults to only test samples
    :param  csv_file: the intermediate CSV file to store the prediction information in;
        will overwrite existing files
    :param device: the device to work on for acquiring the model output
    """
    splits = splits or ('test',)
    if os.path.exists(dest_root):
        raise FileExistsError("dest_root {} exists!".format(dest_root))

    # collect predictions and save to intermediate .csv
    preds = get_model_results(model, dataset_root, save_as=csv_file, device=device, splits=splits)

    # select closest to decision boundary and save into dest_root
    preds = preds[preds.split.isin(splits)]
    pos_pred = select_by_decision_boundary(preds[preds.pred > 0.5], num_imgs_per_cls)
    neg_pred = select_by_decision_boundary(preds[preds.pred <= 0.5], num_imgs_per_cls)

    # save to dest_root
    for img_rel_fp in [*pos_pred, *neg_pred]:
        dest: str = os.path.join(dest_root, img_rel_fp)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(os.path.join(dataset_root, img_rel_fp), dest)


NUM_IMGS_PER_CLS: int = 50

if __name__ == '__main__':
    # region SETTINGS
    # ---------------
    PROJECT_ROOT = "."  # assume that the script is called from project root
    model_pkl_file = os.path.join(PROJECT_ROOT, "alexnet_finetuned.pkl")
    MODEL = model_loaders.modified_alexnet(torch.load(model_pkl_file))
    DEVICE = 'cuda'
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "fasseg", "picasso_dataset")
    DEST_ROOT = os.path.join(
        PROJECT_ROOT, "dataset", "{}_ilp_samples".format(model.model_id(model_name="AlexNet",
                                                                        model_pkl_file=model_pkl_file)))
    CSV_FILE = os.path.join(PROJECT_ROOT, "models",
                            '{}_preds_test.csv'.format(MODEL.__class__.__name__.lower()))
    # endregion

    create_samples_folder(
        model=MODEL,
        dataset_root=DATASET_ROOT,
        dest_root=DEST_ROOT,
        num_imgs_per_cls=NUM_IMGS_PER_CLS,
        csv_file=CSV_FILE,
        device=DEVICE,
        splits=('test',)
    )
