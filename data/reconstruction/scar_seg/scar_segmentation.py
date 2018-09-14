"""
Scar segmentation dataset

Code adapted from Jo Schlemper
"""
import glob
import logging
import os
import re
from itertools import chain

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

from data.transform_wrappers import get_rec_transform, get_rec_seg_transform

DATASET_DIR = 'scar_segmentation'
IMAGE_PATH = 'Analyze/LGE.img'
LABEL_PATH = 'ManualSegmentation/ROI_1_M_MSP_New2.img'

IMAGE_SIZE = 512
NUM_SLICES = 52
DEFAULT_SPLIT_RATIO = [4, 1, 1]

# RegExp to extract case name and slice index from name
# Example name: c11_post_slice01
_CASE_REGEXP = re.compile(r'(c\d+\_(pre|post))\_slice(\d+)')

_TRANSFORM_BY_DATASET_MODE = {
    'reconstruction': get_rec_transform,
    'segmentation': get_rec_seg_transform
}

# Predetermined split for the split ratio [4, 1, 1]
_STATIC_SPLIT = {
    'train': [
        'c03_pre',
        'c43_pre',
        'c47_post',
        'c45_post',
        'c24_post',
        'c13_pre',
        'c49_post',
        'c41_post',
        'c46_pre',
        'c26_pre',
        'c46_post',
        'c38_post',
        'c11_post',
        'c20_post',
        'c19_post',
        'c09_post',
        'c26_post',
        'c37_post',
        'c44_post',
        'c25_post',
        'c02_post',
        'c25_pre',
        'c28_post',
        'c34_pre',
    ],
    'val': [
        'c18_pre',
        'c34_post',
        'c54_pre',
        'c17_post',
        'c18_post',
        'c03_post'
    ],
    'test': [
        'c43_post',
        'c29_post',
        'c44_pre',
        'c13_post',
        'c45_pre',
        'c48_post',
        'c36_post'
    ]
}

assert set(_STATIC_SPLIT['train']).isdisjoint(set(_STATIC_SPLIT['val']))
assert set(_STATIC_SPLIT['train']).isdisjoint(set(_STATIC_SPLIT['test']))
assert set(_STATIC_SPLIT['val']).isdisjoint(set(_STATIC_SPLIT['test']))


def _load_label(image_folder):
    label_path = os.path.join(image_folder, LABEL_PATH)
    label = np.squeeze(nib.load(label_path).get_data())
    return label


def _load_image_and_label(image_folder):
    image_path = os.path.join(image_folder, IMAGE_PATH)
    label_path = os.path.join(image_folder, LABEL_PATH)

    image = np.squeeze(nib.load(image_path).get_data())
    label = np.squeeze(nib.load(label_path).get_data())
    return image, label


def _load_datasets(image_folders, mode='2d', downsize=1, nz=NUM_SLICES):
    assert len(image_folders) > 0
    images = []
    labels = []
    image_ids = []
    for i, image_folder in enumerate(image_folders):
        image, label = _load_image_and_label(image_folder)

        # downsize in k-space
        # image[::downsize,::downsize,:nz]
        # image[::downsize,::downsize,:nz]
        images.append(image[..., :nz])
        labels.append(label[..., :nz])

        for sl in range(nz):
            image_id = '{}_slice{}'.format(os.path.basename(image_folder), sl)
            image_ids.append(image_id)

        # if i > 2:
        #     break

    images = np.array(images)
    labels = np.array(labels)

    if mode == '2d':
        n, nx, ny, nz = images.shape
        images = images.transpose((0, 3, 1, 2)).reshape(-1, nx, ny, 1)
        labels = labels.transpose((0, 3, 1, 2)).reshape(-1, nx, ny, 1)

    return images, labels, image_ids


def _split_data(data_dir, ratio=DEFAULT_SPLIT_RATIO, static_split=True):
    """
    Split dataset into train, validation and test splits
    """
    if static_split:
        train_paths = [os.path.join(data_dir, patient_id)
                       for patient_id in _STATIC_SPLIT['train']]
        val_paths = [os.path.join(data_dir, patient_id)
                     for patient_id in _STATIC_SPLIT['val']]
        test_paths = [os.path.join(data_dir, patient_id)
                      for patient_id in _STATIC_SPLIT['test']]
        for patient_path in chain(train_paths, val_paths, test_paths):
            assert (os.path.isfile(os.path.join(patient_path, IMAGE_PATH)) and
                    os.path.isfile(os.path.join(patient_path, LABEL_PATH))), \
                'Did not find image or label for {}'.format(patient_path)
    else:
        ratio = np.array(ratio, float) / sum(ratio)

        # Get patients directories from dataset
        patient_paths = []
        folder_names = glob.glob(os.path.join(data_dir, 'c*'))

        for folder_name in sorted(folder_names):
            if os.path.isfile(os.path.join(folder_name, IMAGE_PATH)) and \
               os.path.isfile(os.path.join(folder_name, LABEL_PATH)):
                patient_paths.append(folder_name)

        # This should be split in terms of id's and pair each pre and post.
        rng = np.random.RandomState(seed=0)
        rng.shuffle(patient_paths)

        # Split
        n = len(patient_paths)
        n_train, n_validate, n_test = map(int, n * ratio)

        train_paths = patient_paths[:n_train]
        val_paths = patient_paths[n_train:n_train + n_validate]
        test_paths = patient_paths[n_train + n_validate:]

    logging.debug(('n_train: {}, n_validate: {}, '
                  'n_test: {}').format(len(train_paths),
                                       len(val_paths),
                                       len(test_paths)))
    return train_paths, val_paths, test_paths


class ReconstructionDataset(Dataset):
    def __init__(self, images, labels, image_ids,
                 transform, mode='reconstruction'):
        super(Dataset, self).__init__()
        assert mode in ('reconstruction', 'segmentation')
        self.images = images
        self.labels = labels
        self.image_ids = image_ids
        self.transform = transform
        self.mode = mode
        self.name = 'ScarSeg'

    def __getitem__(self, index):
        image = self.images[index]
        if self.mode != 'reconstruction':
            label = self.labels[index]
            # Transform returns:
            # image[0:2]: downsampled image
            # image[2:4]: k-space image
            # image[4:6]: k-space sampling mask
            # image[6:8]: ground truth image
            # label     : segmentation mask
            image, label = self.transform([image, label])
            label = label.squeeze(0).astype(np.int64)
        else:
            # Transform returns:
            # image[0:2]: downsampled image
            # image[2:4]: k-space image
            # image[4:6]: k-space sampling mask
            # image[6:8]: ground truth image
            image = self.transform(image)

        if self.mode == 'reconstruction':
            return {
                'inp': image[0:2],
                'kspace': image[2:4],
                'mask': image[4:6],
                'target': image[6:8]
            }
        elif self.mode == 'segmentation':
            # Segmentation from ground truth reconstructions
            return {
                'inp': image[6:8],
                'target': label
            }

    def __len__(self):
        return len(self.images)

    def get_filename(self, index):
        return self.image_ids[index]

    @staticmethod
    def get_case_and_slice(name):
        m = _CASE_REGEXP.match(name)
        assert m is not None, name
        return m.group(1), m.group(3)


def get_train_set(conf, data_dir):
    dataset_path = os.path.join(data_dir, DATASET_DIR)
    split_ratio = conf.get_attr('split_ratio', default=DEFAULT_SPLIT_RATIO)
    static_split = not conf.get_attr('random_split', default=False)
    image_paths = _split_data(dataset_path, split_ratio, static_split)[0]

    downscale_factor = conf.get_attr('downscale', default=1)
    input_mode = conf.get_attr('input_mode', default='2d')
    images, labels, image_ids = _load_datasets(image_paths, mode=input_mode,
                                               downsize=downscale_factor)

    dataset_mode = conf.get_attr('dataset_mode', default='reconstruction')
    transform_getter = _TRANSFORM_BY_DATASET_MODE[dataset_mode]
    transform = transform_getter(conf, 'train', image_size=IMAGE_SIZE)

    return ReconstructionDataset(images, labels, image_ids,
                                 transform, dataset_mode)


def _get_test_or_val_set(conf, data_dir, fold_idx):
    dataset_path = os.path.join(data_dir, DATASET_DIR)
    split_ratio = conf.get_attr('split_ratio', default=DEFAULT_SPLIT_RATIO)
    static_split = not conf.get_attr('random_split', default=False)
    image_paths = _split_data(dataset_path,
                              split_ratio,
                              static_split)[fold_idx]

    downscale_factor = conf.get_attr('downscale', default=1)
    input_mode = conf.get_attr('input_mode', default='2d')
    images, labels, image_ids = _load_datasets(image_paths, mode=input_mode,
                                               downsize=downscale_factor)

    dataset_mode = conf.get_attr('dataset_mode', default='reconstruction')
    transform_getter = _TRANSFORM_BY_DATASET_MODE[dataset_mode]
    transform = transform_getter(conf, 'test',
                                 image_size=IMAGE_SIZE,
                                 num_images=len(images))

    return ReconstructionDataset(images, labels, image_ids,
                                 transform, dataset_mode)


def get_val_set(conf, data_dir):
    return _get_test_or_val_set(conf, data_dir, fold_idx=1)


def get_test_set(conf, data_dir):
    return _get_test_or_val_set(conf, data_dir, fold_idx=2)

