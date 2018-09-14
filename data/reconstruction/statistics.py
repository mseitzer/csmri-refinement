"""Implementations of metrics for evaluation"""

from collections import OrderedDict
import logging

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from skimage.measure import compare_psnr, compare_ssim, regionprops

from .io import (CASE_KEY, SLICE_KEY, PRED_KEY, TARGET_KEY,
                 LABEL_KEY, TARGET_LABEL_KEY, HEADER_KEY,
                 maybe_convert_to_magnitude)

def _get_index_key(data):
  if SLICE_KEY in data:
    # Computation of metric is per slice
    index_key = '{}_{:02d}'.format(data[CASE_KEY], data[SLICE_KEY])
  else:
    # Computation of metric is per volume
    index_key = data[CASE_KEY]

  return index_key


def compute_psnr(dataset):
  values = OrderedDict()
  for data in dataset:
    pred = maybe_convert_to_magnitude(data[PRED_KEY])
    target = maybe_convert_to_magnitude(data[TARGET_KEY])

    index_key = _get_index_key(data)
    value = compare_psnr(target, pred, data_range=target.max())
    values[index_key] = value

  return pd.Series(values)


def compute_ssim(dataset):
  values = OrderedDict()
  for data in dataset:
    pred = maybe_convert_to_magnitude(data[PRED_KEY]).squeeze()
    target = maybe_convert_to_magnitude(data[TARGET_KEY]).squeeze()

    index_key = _get_index_key(data)
    # Settings to match the original SSIM publication
    value = compare_ssim(target, pred, data_range=target.max(),
                         gaussian_weights=True, sigma=1.5,
                         use_sample_covariance=False)
    values[index_key] = value

  return pd.Series(values)


def compute_seg_score(dataset, seg_score):
  from torch import Tensor
  from utils import cudaify, make_variables

  values = OrderedDict()
  for data in dataset:
    pred = maybe_convert_to_magnitude(data[PRED_KEY])
    pred = Tensor(pred).unsqueeze(0)
    target = Tensor(data[TARGET_LABEL_KEY]).unsqueeze(0)

    pred, target = make_variables((pred, target), volatile=True)
    if seg_score.cuda != '':
      pred, target = cudaify((pred, target))

    index_key = _get_index_key(data)
    value = seg_score(pred, target)
    values[index_key] = value

  return pd.Series(values)


def _dice(prediction, target, class_idx, absent_value):
    A = (prediction.squeeze() == class_idx)
    B = (target.squeeze() == class_idx)

    denom = np.sum(A) + np.sum(B)
    if denom == 0.:
        # Class does not show up in image and predicted this correctly
        return absent_value
    else:
        return 2. * np.sum(A * B) / denom


def compute_dice_scores(dataset, num_classes, absent_value=0.0):
  values_per_class = [OrderedDict() for _ in range(num_classes)]
  for data in dataset:
    pred = data[LABEL_KEY]
    target = data[TARGET_LABEL_KEY]

    index_key = _get_index_key(data)
    for class_idx in range(num_classes):
      value = _dice(pred, target, class_idx, absent_value)
      values_per_class[class_idx][index_key] = value

  return [pd.Series(values, name='dice_class_{}'.format(class_idx))
          for class_idx, values in enumerate(values_per_class)]


def compute_wilcoxon(series1, series2):
  return wilcoxon(series1, series2)
