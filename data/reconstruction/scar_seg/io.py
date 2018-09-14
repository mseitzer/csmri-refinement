import glob
import os

import numpy as np
from scipy.io import loadmat

from ..io import (INPUT_KEY, PRED_KEY, TARGET_KEY,
                  TARGET_LABEL_KEY, IMAGE_KEYS,
                  CaseDataset, load_from_raw)
from .scar_segmentation import ReconstructionDataset


def load_from_jo_format(filepath, pred_key):
  def complex_to_two_channels(x):
    return np.stack((np.real(x), np.imag(x)))

  assert pred_key is not None, \
      'Need prediction key when loading from Jo format'
  mat = loadmat(filepath)

  data = []
  for name, inp, pred, target in zip(mat['slice_names'],
                                     mat['seq_und'].transpose((2, 0, 1)),
                                     mat[pred_key].transpose((2, 0, 1)),
                                     mat['seq_gnd'].transpose((2, 0, 1))):
    name = str(name[0][0][0])
    case, slice_idx = ReconstructionDataset.get_case_and_slice(name)

    data.append({
        'case': case,
        'slice': int(slice_idx),
        INPUT_KEY: complex_to_two_channels(inp),
        PRED_KEY: complex_to_two_channels(pred),
        TARGET_KEY: complex_to_two_channels(target)
    })

  return data


def load_dataset(path, only_load_keys=IMAGE_KEYS,
                 data_format='default', pred_key=None):
  files = sorted(glob.glob(os.path.join(path, '*.mat')))

  data = []
  if data_format == 'jo':
    for file in files:
      data += load_from_jo_format(file, pred_key)
  else:
    for file in files:
      data.append(load_from_raw(file, only_load_keys))

  return data


def load_gt_label(path):
  from .scar_segmentation import (NUM_SLICES, _load_label)
  labels = _load_label(path)[..., :NUM_SLICES]

  res_labels = []
  for slice_idx in range(labels.shape[-1]):
    label = np.expand_dims(labels[:, :, slice_idx], axis=0)
    label = np.ceil(label).astype(np.uint8)
    res_labels.append(label)

  return res_labels


def add_gt_labels(dataset, dataset_path, fold):
  from .scar_segmentation import _split_data

  dataset = CaseDataset(dataset)
  _, val_paths, test_paths = _split_data(dataset_path,
                                         static_split=True)

  fold = val_paths if fold == 'val' else test_paths
  for i, image_folder in enumerate(fold):
    case = os.path.basename(image_folder)
    labels = load_gt_label(image_folder)

    for slice_idx, label in enumerate(labels):
      data = dataset.get_data(case, slice_idx)
      if data is None:
        print(('Warning: did not find {}, slice {} but they '
               'are in GT.').format(case, slice_idx))
        continue

      data[TARGET_LABEL_KEY] = label
      data['has_class_1'] = np.any(label == 1)
