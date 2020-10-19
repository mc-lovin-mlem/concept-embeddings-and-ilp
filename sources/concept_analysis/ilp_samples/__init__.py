"""Utility functions to pick, generate, and load samples for ILP from concept analysis results.
ILP samples currently have the format described in generation.generate:
.npz files holding the float prediction of the main model and masks as numpy arrays for each part.

Typical workflow:
- To collect and copy samples valuable for ILP (close to model decision boundary) into their own
  samples folder structure, use create_samples_folder.
- To generate an <img>.npz file for all <img>.png files in a folder, use generate.
- To load the information of all <img>.npz files from a folder structure into a DataFrame,
  use load.
  The folder structure is a root folder containing 'pos' and 'neg' sub-folders for the
  positive/negative ground truth samples.
"""

from .generation import generate
from .generation import load
from .picking import create_samples_folder
