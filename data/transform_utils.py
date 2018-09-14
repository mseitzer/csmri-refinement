import random

import numpy as np


def normalize_image_array(img):
  """Normalize image by mean and variance

  Mean and variance are computed per channel

  Parameters
  ----------
  img : np.ndarray
    Array of shape CxHxW to normalize

  Returns
  -------
  Normalized array with zero mean and variance one
  """
  mean = np.mean(img, axis=(1, 2), keepdims=True)
  std = np.std(img, axis=(1, 2), keepdims=True)
  return (img - mean) / std


def scale_by_min_max(img):
  """Scale image to have range (0, 1)

  Scaling is based on per-channel minimum and maximum

  Parameters
  ----------
  img : np.ndarray
    Array of shape CxHxW to scale
  """
  img = img - np.min(img, axis=(1, 2), keepdims=True)
  maximum = np.max(img, axis=(1, 2), keepdims=True)
  maximum[maximum == 0] = 1  # Guard against divide-by-zero on uniform images
  return img / maximum


def softmax(logits, axis=0):
  """Computes softmax function along an axis

  Parameters
  ----------
  logits : np.ndarray
    Array of arbitrary shape on which to compute softmax
  axis : int
    Axis along which to compute softmax
  """
  normalized_logits = logits - np.max(logits, axis=axis, keepdims=True)
  probs = np.exp(normalized_logits) / np.sum(np.exp(normalized_logits),
                                             axis=axis, keepdims=True)
  return probs
