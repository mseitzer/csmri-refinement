import torch
from torch.autograd import Variable

def normalize_range(tensor, source_range, clamp=True):
    """Scales tensor from `source_range` to (0, 1) range"""
    tensor = (tensor - source_range[0]) / (source_range[1] - source_range[0])
    if clamp:
        tensor = tensor.clamp(source_range[0], source_range[1])
    return tensor


def scale_to_range(tensor, target_range, clamp=True):
  """Scales tensor from (0, 1) range to `target range`"""
  tensor = tensor * (target_range[1] - target_range[0]) + target_range[0]
  if clamp:
    tensor = tensor.clamp(target_range[0], target_range[1])
  return tensor


def scale_batch_per_example(tensor):
  """Scale batch of tensors to range (0, 1)

  Scaling is done by taking the minimum and maximum per channel and example

  Parameters
  ----------
  tensor : torch.Tensor or torch.autograd.Variable
    Tensor of shape BxCxHxW
  """
  b, c, h, w = tensor.shape
  out = tensor.view(b, c, h * w)  # Reshape so that we only need one min/max op
  out = out - torch.min(out, dim=-1, keepdim=True)[0]
  out = out / torch.max(out, dim=-1, keepdim=True)[0]
  return out.clamp(0., 1.).view(b, c, h, w)


def normalize_batch_per_example(tensor,
                                normalize_mean=True,
                                normalize_std=True):
  """Normalize batch of tensors by mean and variance

  Mean and variance are computed per channel and example

  Parameters
  ----------
  tensor : torch.Tensor or torch.autograd.Variable
    Tensor of shape BxCxHxW
  norm_mean : bool
    If true, subtract per channel mean
  norm_std : bool
    If true, divide by per channel standard deviation
  """
  b, c, h, w = tensor.shape
  out = tensor.view(b, c, h * w)  # Reshape so that we only need one mean op
  if normalize_mean:
    out = out - torch.mean(out, dim=1, keepdim=True)
  if normalize_std:
    out = out / torch.std(out, dim=1, keepdim=True)
  return out.view(b, c, h, w)


def complex_abs(tensor):
  """Compute absolute value of complex image tensor

  Parameters
  ----------
  tensor : torch.Tensor
    Tensor of shape (batch, 2, height, width)

  Returns
  -------
  Tensor with magnitude image of shape (batch, 1, height, width)
  """
  tensor = (tensor[:, 0] ** 2 + tensor[:, 1] ** 2) ** 0.5
  return torch.unsqueeze(tensor, dim=1)


def magnitude_image(tensor):
  """Compute magnitude image of complex image tensor and scales range
  to be between (0, 1) using minimum and maximum of each image

  Parameters
  ----------
  tensor : torch.Tensor
    Tensor of shape (batch, 2, height, width)

  Returns
  -------
  Tensor with magnitude image of shape (batch, 1, height, width)
  """
  tensor = complex_abs(tensor)

  b, c, h, w = tensor.shape
  tensor = tensor.view(b, c, h * w)
  minimum, _ = tensor.min(dim=2, keepdim=True)
  tensor = tensor - minimum
  maximum, _ = tensor.max(dim=2, keepdim=True)
  tensor = tensor / maximum  # tensor has range (0, 1)
  return tensor.view(b, c, h, w)


def convert_to_one_hot(tensor, num_classes=None):
  """Convert dense classification targets to one-hot representation

  Parameters
  ----------
  tensor : torch.Tensor
    Tensor of dimensionality N
  num_classes : int
    Number of entries C of the one-hot representation. If None, use maximum
    value of the tensor

  Returns
  -------
  Tensor of with dimensionality N+1, where the last dimension has shape C
  """
  t = tensor.type(torch.LongTensor).view(-1, 1)
  if num_classes is None:
    num_classes = int(torch.max(t)) + 1

  t_one_hot = torch.zeros(t.size()[0], num_classes).scatter_(1, t, 1)
  t_one_hot = t_one_hot.view(*tensor.shape, -1)
  return t_one_hot


def print_tensor_stats(t, prefix='', debug=False):
  import logging
  tmin = float(t.min())
  tmax = float(t.max())
  tmean = float(t.mean())
  tstd = float(t.std())
  tmedian = float(t.contiguous().median())
  s = '{}: Min: {:.9f}, Max: {:.9f}, Avg: {:.9f}, Std: {:.9f}, Median: {:.9f}'
  s = s.format(prefix, tmin, tmax, tmean, tstd, tmedian)
  if debug:
    logging.debug(s)
  else:
    logging.info(s)
