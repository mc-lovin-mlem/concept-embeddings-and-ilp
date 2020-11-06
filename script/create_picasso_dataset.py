"""Script to generate the picasso dataset from FASSEG samples.

The FASSEG V1 dataset is a collection of frontal face portraits with segmentation of facial
features. Here of relevance are left and right eyes, nose, mouth.

The picasso dataset is a dataset of constructed face-like images with binary labels of
positive=valid face and negative=invalid face.
The images are constructed of blank faces (facial features manually removed), to which features
cropped from FASSEG images are added.
The features are positioned with some noise around the standard positions of the features
-- for negative samples however, the features may be randomly shuffled, e.g. the nose is where
usually an eye would be.
The labels are invariant to swapping of eyes (it is not recorded whether an eye originally was at
the right or left position in the image), so images where only eyes are swapped are labeled
positive.

The dataset is saved to PICASSO_ROOT.
"""

import copy
import os
from os import listdir
from typing import List, Optional, Tuple, Sequence

import numpy as np
from skimage import io, transform, img_as_ubyte
from tqdm import tqdm

PROJECT_ROOT = "."  # assume the script is called from within project root
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
ORIGS_PATH = os.path.join(DATASET_ROOT, "fasseg", "heads_original")
NO_FACES_PATH = os.path.join(PROJECT_ROOT, "dataset", "fasseg", "heads_no_face")
LABELS_PATH = os.path.join(PROJECT_ROOT, "dataset", "fasseg", "heads_labels")

PICASSO_ROOT = os.path.join(DATASET_ROOT, 'picasso_dataset')
"""Path to the root directory into which to save the generated images."""

DATASET_SIZE = 500
CREATE_MASKS = True

origs_files = listdir(ORIGS_PATH)
no_faces_files = listdir(NO_FACES_PATH)
labels_files = listdir(LABELS_PATH)

data_chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
"""To become list of tuples (original image, image with face removed, segmentation of orig image)"""

for f in origs_files:
    orig_img = io.imread(os.path.join(ORIGS_PATH, f))
    no_face_img = io.imread(os.path.join(NO_FACES_PATH, f))
    label_img = io.imread(os.path.join(LABELS_PATH, f))
    data_chunks.append(tuple((orig_img, no_face_img, label_img)))


left_eyes_pixels = []
right_eyes_pixels = []
nose_pixels = []
mouth_pixels = []
left_eye_orig_positions = []
right_eye_orig_positions = []
nose_orig_positions = []
mouth_orig_positions = []
for dc in data_chunks: #go over all images
    #left eye
    indices = np.where(np.all(dc[2] == (0, 0, 255), axis=-1))
    pixels = dc[0][indices]
    min_row = np.min(indices[0])
    min_col = np.min(indices[1])
    left_eye_orig_positions.append(tuple((min_row, min_col)))
    for i in range(len(indices[0])):
        indices[0][i] -= min_row
    for i in range(len(indices[1])):
        indices[1][i] -= min_col
    left_eyes_pixels.append(tuple((indices, pixels)))

    #right eye
    indices = np.where(np.all(dc[2] == (255, 0, 255), axis=-1))
    pixels = dc[0][indices]
    min_row = np.min(indices[0])
    min_col = np.min(indices[1])
    right_eye_orig_positions.append(tuple((min_row, min_col)))
    for i in range(len(indices[0])):
        indices[0][i] -= min_row
    for i in range(len(indices[1])):
        indices[1][i] -= min_col
    right_eyes_pixels.append(tuple((indices, pixels)))

    #nose
    indices = np.where(np.all(dc[2] == (0, 255, 255), axis=-1))
    pixels = dc[0][indices]
    min_row = np.min(indices[0])
    min_col = np.min(indices[1])
    nose_orig_positions.append(tuple((min_row, min_col)))
    for i in range(len(indices[0])):
        indices[0][i] -= min_row
    for i in range(len(indices[1])):
        indices[1][i] -= min_col
    nose_pixels.append(tuple((indices, pixels)))

    #mouth
    indices = np.where(np.all(dc[2] == (0, 255, 0), axis=-1))
    pixels = dc[0][indices]
    min_row = np.min(indices[0])
    min_col = np.min(indices[1])
    mouth_orig_positions.append(tuple((min_row, min_col)))
    for i in range(len(indices[0])):
        indices[0][i] -= min_row
    for i in range(len(indices[1])):
        indices[1][i] -= min_col
    mouth_pixels.append(tuple((indices, pixels)))


def generate_image(canvas: int, organ_mapping: Sequence[Sequence[str]]) -> np.ndarray:
    """
    :param canvas: the index of the faceless images to take as background like 0, 1, 2, etc..
    :param organ_mapping: a list of lists like ['nose', 'mouth'] indicating that there should be
        a mouth on the position of the nose
    :return:
    """
    frankenstein_image: np.ndarray = copy.deepcopy(data_chunks[canvas][1])
    mask: np.ndarray = np.zeros(shape=frankenstein_image.shape)

    original_positions = None
    pixel_info = None
    mask_color = None
    output = None
    output_mask = None

    for om in organ_mapping:
        if om[0] == 'left_eye':
            original_positions = left_eye_orig_positions
        elif om[0] == 'right_eye':
            original_positions = right_eye_orig_positions
        elif om[0] == 'nose':
            original_positions = nose_orig_positions
        elif om[0] == 'mouth':
            original_positions = mouth_orig_positions
        else:
            raise ValueError("Wrong 'From' mapping {} given!".format(om[0]))

        if om[1] == 'left_eye':
            pixel_info = left_eyes_pixels
            mask_color = (1.0, 0.0, 0.0)
        elif om[1] == 'right_eye':
            pixel_info = right_eyes_pixels
            mask_color = (1.0, 0.0, 0.0)
        elif om[1] == 'nose':
            pixel_info = nose_pixels
            mask_color = (0.0, 1.0, 0.0)
        elif om[1] == 'mouth':
            pixel_info = mouth_pixels
            mask_color = (0.0, 0.0, 1.0)
        else:
            raise ValueError("Wrong 'To' mapping {} given!".format(om[1]))

        flag = True
        while flag:
            try:
                flag = False
                organ_donor: int = np.random.randint(6)
                delta_row: int = np.random.randint(10)
                delta_col: int = np.random.randint(10)
                new_organ_indices: List[Optional[int]] = [None] * 2
                new_organ_indices[0] = pixel_info[organ_donor][0][0] + original_positions[canvas][0] + delta_row
                new_organ_indices[1] = pixel_info[organ_donor][0][1] + original_positions[canvas][1] + delta_col
                frankenstein_image[tuple(new_organ_indices)] = 0.2 * frankenstein_image[tuple(new_organ_indices)] + (1.0-0.2) * pixel_info[organ_donor][1]
                io.imshow(frankenstein_image)
                io.show()
                mask[tuple(new_organ_indices)] = mask_color
            except IndexError as error:
                flag = True

        height = frankenstein_image.shape[0]
        width = frankenstein_image.shape[1]
        bigger = max(height, width)
        smaller = min(height, width)
        adjusted_img: np.ndarray = np.zeros((bigger, bigger, 3))
        adjusted_mask: np.ndarray = np.zeros((bigger, bigger, 3))
        offset = int((bigger/2)-(smaller/2))
        adjusted_img[:, offset:offset+smaller, :] = frankenstein_image
        adjusted_mask[:, offset:offset+smaller, :] = mask
        output: np.ndarray = transform.resize(adjusted_img, (224, 224))
        output_mask: np.ndarray = transform.resize(adjusted_mask, (224, 224))


    return output, output_mask

#pos
for i in tqdm(range(DATASET_SIZE//2)):
    # if i % 100 == 0:
    #     print(i, "/", DATASET_SIZE//2)
    canvas = np.random.randint(6)
    gi, gm = generate_image(canvas, [['left_eye', 'left_eye'], ['right_eye', 'right_eye'], ['nose', 'nose'], ['mouth', 'mouth']])
    gi = gi.astype(float)
    gi /= 255.0
    if i <= int(DATASET_SIZE//2 * 0.9):
        split = "train"
    else:
        split = "test"
    im_path = os.path.join(PICASSO_ROOT, split, "pos", "pic_" + str(i).zfill(5) + ".png")
    # Create directory if it does not yet exist:
    os.makedirs(os.path.dirname(im_path), exist_ok=True)
    io.imsave(im_path, img_as_ubyte(gi))

    mask_path = os.path.join(PICASSO_ROOT, split, "pos", "mask_" + str(i).zfill(5) + ".png")
    # Create directory if it does not yet exist:
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    io.imsave(mask_path, img_as_ubyte(gm))

#neg
for i in tqdm(range(DATASET_SIZE//2)):
    # if i % 100 == 0:
    #     print(i, "/", DATASET_SIZE//2)
    canvas = np.random.randint(6)
    organs = ['left_eye', 'right_eye', 'nose', 'mouth']
    while (organs[0] == 'left_eye' and organs[1] == 'right_eye' and organs[2] == 'nose' and organs[3] == 'mouth') or (organs[0] == 'right_eye' and organs[1] == 'left_eye'):
        np.random.shuffle(organs)
    gi, gm = generate_image(canvas, [['left_eye', organs[0]], ['right_eye', organs[1]], ['nose', organs[2]], ['mouth', organs[3]]])
    gi = gi.astype(float)
    gi /= 255.0
    if i <= int(DATASET_SIZE//2 * 0.9):
        split = "train"
    else:
        split = "test"
    im_path = os.path.join(PICASSO_ROOT, split, "neg", "pic_" + str(i).zfill(5) + ".png")
    # Create directory if it does not yet exist:
    os.makedirs(os.path.dirname(im_path), exist_ok=True)
    io.imsave(im_path, img_as_ubyte(gi))

    mask_path = os.path.join(PICASSO_ROOT, split, "neg", "mask_" + str(i).zfill(5) + ".png")
    # Create directory if it does not yet exist:
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    io.imsave(mask_path, img_as_ubyte(gm))
