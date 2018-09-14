import numpy as np
import torch.nn.functional as F

from data.transform_utils import normalize_image_array


def input_transform(normalize=False, scale_by_max=False):
  """Build input transform for segmentation

  Transform expects tuple (input, target) as input, where:
    - input: image of shape (height, width, depth)
    - target: dense segmentation mask of shape (height, width, 1)

  Transform returns tuple (input, target), where:
    - input: image of shape (depth, height, width)
    - target: dense segmentation mask of shape (height, width)

  Parameters
  ----------
  normalize : bool
    If true, normalize each channel of input image to have zero mean and unit
    variance
  scale_by_max : bool
    If true, scale each channel to range (0, 1) by dividing by the per-channel
    maximum
  """
  def transform(inp, target):
    inp = inp.transpose((2, 0, 1)).astype(np.float32)
    if normalize:
      inp = normalize_image_array(inp)
    if scale_by_max:
      inp /= np.max(inp, axis=0, keepdims=True) + 1e-9
      inp = inp.clip(min=0, max=1)

    target = target.squeeze(2).astype(np.int64)
    return inp, target

  return transform


def output_transform():
  def transform(pred, target):
    probs = F.softmax(pred, dim=1)
    _, predicted_classes = probs.max(dim=1)
    return predicted_classes.unsqueeze(dim=1), target.unsqueeze(dim=1)

  return transform
