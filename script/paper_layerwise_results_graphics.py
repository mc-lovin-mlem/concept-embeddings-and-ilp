#!/usr/bin/env python
"""Code to produce figure on layer-wise concept embedding results used in the paper.
Call from within project root."""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# Settings
# ===============
parser = argparse.ArgumentParser(description="Create a figure depicting layer-wise concept embedding "
                                             "results from an concept analysis experiment folder.")
parser.add_argument('--resnext_exp', type=str,
                    help='Experiment folder name of the ResNeXt experiment to use results of.')
parser.add_argument('--alexnet_exp', type=str,
                    help='Experiment folder name of the ResNeXt experiment to use results of.')
parser.add_argument('--vgg16_exp', type=str,
                    help='Experiment folder name of the ResNeXt experiment to use results of.')
parser.add_argument('--analysis_root', type=str,
                    default=os.path.join("experiments", "concept_analysis"),
                    help='Root under which to find all the analysis experiment folders.')
parser.add_argument('--dest_fp', type=str,
                    default=os.path.join("paper-tex", "figures", "layerwise_results_concept_embedding.pdf"),
                    help='File path to which to store the resulting graphics. '
                         'The ending determines the format.')

args = parser.parse_args()
exp_folders = dict(
    ResNeXt=args.resnext_exp,
    AlexNet=args.alexnet_exp,
    VGG16=args.vgg16_exp,
)
CONCEPTS = ['NOSE', 'MOUTH', 'EYES']
LAYER_INDEX_OFFSETS = dict(ResNeXt=4, AlexNet=2, VGG16=5)  # the first layer occuring in stats is index ...
exp_roots = {m_name: os.path.join(args.analysis_root, d) for m_name, d in exp_folders.items()}


# Stats loading and extraction of (mean & standard deviation of) set IoU values:
# ===============================================================================
def load_stats(file_name, verbose=True, mean_label='mean set IoU'):
    """Load the (mean) set IoU values (and standard deviations) for each layer from the stats file."""
    c_stats = {}
    for model in exp_folders:
        c_stats[model] = {}
        for concept in CONCEPTS:
            exp_root = exp_roots[model]
            stats = pd.read_csv(os.path.join(exp_root, concept, file_name))
            stats.rename(columns={'Unnamed: 0': 'layer', 'Unnamed: 1': 'run'}, inplace=True)
            layers = sorted(stats['layer'].unique())
            layer_idxs = {l: i + LAYER_INDEX_OFFSETS[model] for i, l in enumerate(layers)}
            stats.loc[:, 'layer_idx'] = stats['layer'].apply(lambda l: layer_idxs[l])

            if verbose:
                print(model, concept, stats)

            # collect mean and standard deviation:
            layer_stats = {layer: stats[stats['layer'] == layer]['test_set_iou'] for layer in layers}
            c_stats[model][concept] = pd.DataFrame({layer_idxs[layer]: {mean_label: layer_stats[layer].mean(),
                                                                        'std.dev': layer_stats[layer].std()}
                                                    for layer in layers}).transpose()
    return c_stats


# Plotting definition for set IoU:
# ====================================

concept_stats = load_stats('stats.csv', verbose=False)
concept_bests = load_stats('best_emb_stats.csv', mean_label='set IoU', verbose=False)


def plot_model_stats(model, ax, plot_bests=False):
    """Plot the model statistics into pyplot axis ax."""
    means = concept_stats[model]
    for concept in CONCEPTS:
        c_means = pd.Series(means[concept]['mean set IoU'], name=concept)
        ax = c_means.plot(yerr=means[concept]['std.dev'], capsize=2, legend=True, ax=ax)
        if plot_bests:
            concept_bests[model][concept]['set IoU'].plot(ax=ax)
        ax.set_title(model)
        ax.set_ylabel('set IoU')
        ax.set_xticks(means[concept].index)
        ax.set_xlabel('conv layer / block index')


# Actual plots:
# ===============

plt.close('all')
fig, axes = plt.subplots(1, len(exp_roots), figsize=(len(exp_roots) * 3.6, 2.1), sharey='row', dpi=100)
fig.tight_layout()
for i, model_name in enumerate(sorted(exp_roots)):
    plot_model_stats(model_name, axes[i], plot_bests=False)
plt.savefig(args.dest_fp, bbox_inches='tight')
