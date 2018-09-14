import torch


def compute_average_dice(prediction, target, num_classes,
                         excluded_class=-1, absent_value=0.0):
  """Computes average dice score between target and prediction

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Predicted segmentation map with densely encoded classes
  target : torch.Tensor or torch.autograd.Variable
    Target segmentation map with densely encoded classes
  num_classes : int
    Number of classes
  excluded_class : int
    Index of a class to exclude, e.g. the background class
  absent_value : float
    Value to return if the class is not present in both target and prediction
  """
  score = 0.0
  for class_idx in range(num_classes):
    if class_idx == excluded_class:
      continue
    A = (prediction == class_idx).cpu().type(torch.LongTensor)
    B = (target == class_idx).cpu().type(torch.LongTensor)

    denom = A.sum().data[0] + B.sum().data[0]
    if denom == 0.0:
      # Class does not show up in image and predicted this correctly
      score += absent_value
    else:
      score += 2 * (A * B).sum().data[0] / denom

  if excluded_class != -1:
    num_classes -= 1
  return score / num_classes


def compute_dice(prediction, target, class_idx, absent_value=0.0):
  """Computes dice score between target and prediction for one class

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Predicted segmentation map with densely encoded classes
  target : torch.Tensor or torch.autograd.Variable
    Target segmentation map with densely encoded classes
  class_idx : int
    Index of class to compute dice for
  absent_value : float
    Value to return if the class is not present in both target and prediction
  """
  A = (prediction == class_idx).cpu().type(torch.LongTensor)
  B = (target == class_idx).cpu().type(torch.LongTensor)

  denom = A.sum().data[0] + B.sum().data[0]
  if denom == 0.0:
    # Class does not show up in image and predicted this correctly
    # Logically, this value should be 1, but ACDC challenge uses 0 here
    return absent_value
  else:
    return 2 * (A * B).sum().data[0] / denom
