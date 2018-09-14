from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

CASE_KEY = 'case'
SLICE_KEY = 'slice'
HEADER_KEY = 'header'

INPUT_KEY = 'input'
PRED_KEY = 'pred'
TARGET_KEY = 'target'
LABEL_KEY = 'label'
TARGET_LABEL_KEY = 'tlabel'


IMAGE_KEYS = [INPUT_KEY, PRED_KEY, TARGET_KEY]
LABEL_KEYS = [LABEL_KEY, TARGET_LABEL_KEY]


class CaseDataset(object):
  def __init__(self, dataset):
    self.slices_by_case = OrderedDict()
    for data in dataset:
      self.slices_by_case.setdefault(data[CASE_KEY], []).append(data)

    for case in self.slices_by_case:
      slices = self.slices_by_case[case]
      self.slices_by_case[case] = sorted(slices, key=lambda s: s[SLICE_KEY])

  def __iter__(self):
    return self.slice_iter()

  def get_data(self, case, slice_idx):
    if case not in self.slices_by_case:
      raise ValueError('Did not find case {}'.format(case))

    slices = self.slices_by_case[case]
    if slice_idx >= len(slices):
      raise ValueError('Slice index {} not existing'.format(slice_idx))

    return slices[slice_idx]

  def get_data_by_name(self, name):
    """Get data by name

    Name is expected to be of the form 'case_slice'
    """
    parts = name.split('_')
    case = '_'.join(parts[:-1])
    slice_idx = int(parts[-1])
    return self.get_data(case, slice_idx)

  def slice_iter(self, only_class_1=False):
    for case, slices in self.slices_by_case.items():
      for data in slices:
        if only_class_1:
          if data.get('has_class_1', False):
            yield data
        else:
          yield data

  def volume_iter(self):
    """Iterates over cases joint slicewise to a volume"""
    for case, slices in self.slices_by_case.items():
      vol_data = {CASE_KEY: case}
      keys = [k for k in slices[0] if isinstance(slices[0][k], np.ndarray)]
      for key in keys:
        vol_data[key] = np.stack((data[key] for data in slices), axis=0)
      yield vol_data


def _cabs(x):
  return (x[0] ** 2 + x[1] ** 2) ** 0.5


def load_from_raw(filepath, only_load_keys=IMAGE_KEYS):
  mat = loadmat(filepath)
  data = {
      CASE_KEY: str(mat[CASE_KEY][0]),
      SLICE_KEY: int(mat[SLICE_KEY][0]),
  }

  if only_load_keys is None:
    return data

  for key in only_load_keys:
    assert key in mat
    data[key] = mat[key]

  return data


def save_raw(filepath, name, inp, prediction, target, dataset):
  def maybe_squeeze_batch_dim(arr, arr_name):
    if len(arr.shape) == 4:
      assert arr.shape[0] == 1, \
          '{} should have batch dimension 1 but has shape {}'.format(arr_name,
                                                                     arr.shape)
      arr = arr.squeeze(axis=0)

    assert len(arr.shape) == 3, \
        ('{} should have 3 dimensions or batch size 1'
         ' but has shape {}').format(arr_name, arr.shape)
    return arr

  assert (isinstance(inp, np.ndarray) and
          isinstance(prediction, np.ndarray) and
          isinstance(target, np.ndarray))

  inp = maybe_squeeze_batch_dim(inp, 'Input')
  prediction = maybe_squeeze_batch_dim(prediction, 'Prediction')
  target = maybe_squeeze_batch_dim(target, 'Target')

  case, slice_idx = dataset.get_case_and_slice(name)

  data = {
      CASE_KEY: case,
      SLICE_KEY: slice_idx,
      INPUT_KEY: inp,
      PRED_KEY: prediction,
      TARGET_KEY: target
  }
  savemat(filepath, data)


def check_integrity(dataset1, dataset2, rtol=1e-05, atol=1e-07):
  for slice1, slice2 in zip(dataset1.slice_iter(), dataset2.slice_iter()):
    assert slice1[CASE_KEY] == slice2[CASE_KEY], \
        '{} vs {}'.format(slice1[CASE_KEY], slice2[CASE_KEY])
    assert slice1[SLICE_KEY] == slice2[SLICE_KEY], \
        '{} vs {}'.format(slice1[CASE_KEY], slice2[CASE_KEY])

    inp1 = maybe_convert_to_magnitude(slice1[INPUT_KEY])
    inp2 = maybe_convert_to_magnitude(slice2[INPUT_KEY])
    target1 = maybe_convert_to_magnitude(slice1[TARGET_KEY])
    target2 = maybe_convert_to_magnitude(slice1[TARGET_KEY])

    if not np.allclose(inp1, inp2, rtol=rtol, atol=atol):
      print('Input not equal')
      return slice1, slice2
    if not np.allclose(target1, target2, rtol=rtol, atol=atol):
      print('Targets not equal')
      return slice1, slice2

  return None


def maybe_convert_to_magnitude(data):
  def maybe_convert(image):
    if image.shape[0] == 2:
      return np.expand_dims(_cabs(image), axis=0)
    return image

  if isinstance(data, np.ndarray):
    return maybe_convert(data)

  data = data.copy()
  for key in IMAGE_KEYS:
    if key in data:
      data[key] = maybe_convert(data[key])

  return data


def prepare_for_visualization(data):
  PERCENTILE_LOW = 0.5
  PERCENTILE_HIGH = 99.5

  def scale(image):
    assert image.shape[0] == 1
    p_low, p_high = np.percentile(image, (PERCENTILE_LOW, PERCENTILE_HIGH))
    # Threshold to within percentiles
    image[image < p_low] = p_low
    image[image > p_high] = p_high
    # Scale image to range (0, 1)
    image = image / p_high - p_low
    return image.squeeze()

  if isinstance(data, np.ndarray):
    return scale(data)

  data = data.copy()
  for key in IMAGE_KEYS:
    if key in data:
      data[key] = scale(data[key])

  for key in LABEL_KEYS:
    if key in data:
      data[key] = data[key].squeeze()

  return data
