import os
import random
import sys
from typing import List

PROJECT_ROOT = os.path.join("..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sources", "concept_analysis", "ext_lib"))
from hybrid_learning.datasets.custom.fasseg import FASSEGHandle


class PicassoMaskHandle(FASSEGHandle):
    """Handle for picasso mask dataset.
    The structure of the dataset is very simple:
    under dataset_root lie sub-folders 'pos' and 'neg' each with both input images and
    same-sized masks. The samples from 'pos' folder belong to positive class, those of 'neg' folder
    to the negative class.
    An input image has file name pic_<ID>.png, and the mask for the input is stored in
    mask_<ID>.png.
    The masks are RGB images with the following encoding (see PICASSO_COLORS):
    - eyes: red
    - nose: green
    - mouth: blue
    """
    PICASSO_COLORS = {
        "EYES": (255, 0, 0),
        "NOSE": (0, 255, 0),
        "MOUTH": (0, 0, 255)
    }
    """Mapping of part names to mask color encoding."""

    def __init__(self, dataset_root: str, part_name: str, restrict_cls_to: str = 'all',
                 **kwargs):
        """
        :param dataset_root: the directory under which to find the images;
        :param part_name: the part to select in the masks; key in PICASSO_COLORS
        :param restrict_cls_to: the classification class to restrict samples to; either 'pos', 'neg', or 'all'
        :param kwargs: parameters for BaseDataset
        """
        super(PicassoMaskHandle, self).__init__(
            # fake dataset root to not raise during value checks:
            dataset_root=os.path.join(dataset_root, 'neg'),
            # fake annotations root to ignore argument:
            annotations_root=os.path.join(dataset_root, 'neg'),
            part=self.PICASSO_COLORS[part_name],
            part_name=part_name,
            **kwargs)
        self.dataset_root = dataset_root
        self.annotations_root = None

        # Select class of images:
        img_fns = {}
        for picasso_cls in ('pos', 'neg'):
            cls_root = os.path.join(self.dataset_root, picasso_cls)
            img_fns[picasso_cls] = [os.path.join(picasso_cls, fn) for fn in os.listdir(cls_root)
                                    if os.path.isfile(os.path.join(cls_root, fn))]
        if not restrict_cls_to or restrict_cls_to == 'all':
            all_img_fns: List[str] = [*img_fns['pos'], *img_fns['neg']]
        elif restrict_cls_to in img_fns.keys():
            all_img_fns = img_fns[restrict_cls_to]
        else:
            raise ValueError('Unknown restrict_cls_to option {}'.format(restrict_cls_to))
        # Discard mask files:
        self.img_fns = [fn for fn in all_img_fns if "pic_" in fn]
        # Shuffle:
        random.shuffle(self.img_fns)

        # Check existence of all masks:
        for i in range(len(self.img_fns)):
            mask_fn = self.mask_filepath(i)
            if not os.path.isfile(mask_fn):
                raise ValueError("Mask {} for image {} not found!".format(mask_fn, self.img_fns[i]))

    def mask_filepath(self, i):
        """Provide the path to the ith mask.
        Warning: For compatibility with the FASSEG dataset, this must also for mask files
        yield a valid filepath (in this case the mask filepath itself)!"""
        mask_fn = self.img_fns[i].replace('pic_', 'mask_')
        return os.path.join(self.dataset_root, mask_fn)
