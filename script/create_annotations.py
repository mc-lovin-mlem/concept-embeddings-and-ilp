import argparse
import os
import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import PIL
from PIL import Image
from scipy.ndimage import label
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm


def load_results(results_root: str, gt_class_dirs=('pos', 'neg')):
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


def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def find_regions(mask, pos_neg, organ, f):
    labels, numL = label(mask)
    label_indices = [(labels == i).nonzero() for i in range(1, numL + 1)]
    cluster_sizes = []
    for li in (range(len(label_indices))):
        cluster_sizes.append(len(label_indices[li][0]))
    if organ == 'EYES':
        argsorted = np.argsort(cluster_sizes)
        if len(cluster_sizes) > 0:
            biggest_cluster_index = argsorted[-1]
            # row, col
            min_row = min(label_indices[biggest_cluster_index][0])
            max_row = max(label_indices[biggest_cluster_index][0])
            min_col = min(label_indices[biggest_cluster_index][1])
            max_col = max(label_indices[biggest_cluster_index][1])
            f.write("biggest_eye\n")
            f.write(str(min_row) + "," + str(max_row) + "," + str(min_col) + "," + str(max_col) + "\n")
            pos_eye_1 = ((max_row + min_row) / 2, (max_col + min_col) / 2)
        else:
            f.write("biggest_eye\n")
            f.write("-1,-1,-1,-1\n")

        if len(cluster_sizes) > 1:
            second_biggest_cluster_index = argsorted[-2]
            min_row = min(label_indices[second_biggest_cluster_index][0])
            max_row = max(label_indices[second_biggest_cluster_index][0])
            min_col = min(label_indices[second_biggest_cluster_index][1])
            max_col = max(label_indices[second_biggest_cluster_index][1])
            f.write("second_eye\n")
            f.write(str(min_row) + "," + str(max_row) + "," + str(min_col) + "," + str(max_col) + "\n")
            pos_eye_2 = ((max_row + min_row) / 2, (max_col + min_col) / 2)
        else:
            f.write("second_eye\n")
            f.write("-1,-1,-1,-1\n")
        # print("Pos_eye_1", pos_eye_1)
        # print("Pos_eye_2", pos_eye_2)
    elif organ == 'NOSE':
        if len(cluster_sizes) > 0:
            argsorted = np.argsort(cluster_sizes)
            biggest_cluster_index = argsorted[-1]
            # row, col
            min_row = min(label_indices[biggest_cluster_index][0])
            max_row = max(label_indices[biggest_cluster_index][0])
            min_col = min(label_indices[biggest_cluster_index][1])
            max_col = max(label_indices[biggest_cluster_index][1])
            f.write("nose\n")
            f.write(str(min_row) + "," + str(max_row) + "," + str(min_col) + "," + str(max_col) + "\n")
            pos_nose = ((max_row + min_row) / 2, (max_col + min_col) / 2)
        else:
            f.write("nose\n")
            f.write("-1,-1,-1,-1\n")
        # print(pos_nose)
    elif organ == 'MOUTH':
        if len(cluster_sizes) > 0:
            argsorted = np.argsort(cluster_sizes)
            biggest_cluster_index = argsorted[-1]
            # row, col
            min_row = min(label_indices[biggest_cluster_index][0])
            max_row = max(label_indices[biggest_cluster_index][0])
            min_col = min(label_indices[biggest_cluster_index][1])
            max_col = max(label_indices[biggest_cluster_index][1])
            f.write("mouth\n")
            f.write(str(min_row) + "," + str(max_row) + "," + str(min_col) + "," + str(max_col) + "\n")
            pos_mouth = ((max_row + min_row) / 2, (max_col + min_col) / 2)
        else:
            f.write("mouth\n")
            f.write("-1,-1,-1,-1\n")


if __name__ == "__main__":
    PROJECT_ROOT = "."  # assume by default the script is called from within project root

    parser = argparse.ArgumentParser(
        description='Generate a file that contains part positions for samples from a concept analysis. '
                    'The resulting files are '
                    'dest_root/annotations_positive.csv and '
                    'dest_root/annotations_negative.csv.')
    parser.add_argument('--dest_root', type=str,
                        default=None,
                        help='The path to the root folder under which to create the .csv files. '
                             'Defaults to experiments/ilp (see also --exp_name).')
    parser.add_argument('--src_root', type=str,
                        default=None,
                        help='The root under which to find the samples and their evaluations selected '
                             'for ILP. Defaults to experiments/ilp_samples (see also --exp_name).')
    parser.add_argument('--exp_name', type=str,
                        default=None,
                        help='If --dest_root or --src_root is not given but --exp_name, the roots default to '
                             'experiments/{ilp_samples,ilp}/exp_name. exp_root is usually the '
                             'folder name of the experiment root folder for the corresponding '
                             'concept analysis.')

    args = parser.parse_args()
    dest_root = args.dest_root
    results_root = args.src_root
    if dest_root is None:
        dest_root = os.path.join(PROJECT_ROOT, "experiments", "ilp")
        if args.exp_name is not None:
            dest_root = os.path.join(dest_root, args.exp_name)
    if results_root is None:
        results_root = os.path.join(PROJECT_ROOT, "experiments", "ilp_samples")
        if args.exp_name is not None:
            results_root = os.path.join(results_root, args.exp_name)

    anno_file_p = open(os.path.join(dest_root, "annotations_positive.csv"), "w")
    anno_file_n = open(os.path.join(dest_root, "annotations_negative.csv"), "w")

    results = load_results(results_root, gt_class_dirs=('pos', 'neg'))

    for s in range(len(results)):
        # print("Sample ", s, "has just started.")

        sample = results.iloc[s]
        pred = sample['pred']
        pred_cls = None
        ground_truth = sample['ground_truth']

        # print("Sample is by ground truth", ground_truth)

        af = None
        if pred > 0.5:
            # print("Sample is predicted positive.")
            pred_cls = 'pos'
            af = anno_file_p
            anno_file_p.write(str(s) + "\n")
        else:
            # print("Sample is predicted negative.")
            pred_cls = 'neg'
            af = anno_file_n
            anno_file_n.write(str(s) + "\n")

        fp = False
        fn = False
        if (pred_cls == 'pos' and ground_truth == 'neg'):
            fp = True
        if (pred_cls == 'neg' and ground_truth == 'pos'):
            fn = True

        nose_mask = sample['NOSE']
        nose_mask = nose_mask.resize((224, 224), resample=Image.BILINEAR)
        nose_mask = np.array(nose_mask)
        mouth_mask = sample['MOUTH']
        mouth_mask = mouth_mask.resize((224, 224), resample=Image.BILINEAR)
        mouth_mask = np.array(mouth_mask)
        eyes_mask = sample['EYES']
        eyes_mask = eyes_mask.resize((224, 224), resample=Image.BILINEAR)
        eyes_mask = np.array(eyes_mask)

        find_regions(np.array(nose_mask), pred_cls, 'NOSE', af)
        find_regions(np.array(mouth_mask), pred_cls, 'MOUTH', af)
        find_regions(np.array(eyes_mask), pred_cls, 'EYES', af)

        if fp:
            print("False Positive at sample", s)
        if fn:
            print("False Negative at sample", s)

    anno_file_p.close()
    anno_file_n.close()
