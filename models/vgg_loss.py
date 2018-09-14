import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tensor_transforms import normalize_range, complex_abs

_CRITERIONS = {
    'MSE': F.mse_loss,
    'L1': F.l1_loss
}


class VGGLoss(nn.Module):
  def __init__(self, loss_name, cuda, blocks=-1, criterion='L1', weights=None):
    super(VGGLoss, self).__init__()

    if loss_name == 'VGG19':
      from models.vgg import VGG19
      module = VGG19
    else:
      raise ValueError('Unknown VGG loss {}'.format(loss_name))

    if blocks == -1:
      blocks = [module.LAST_FEATURE_MAP]
    elif not isinstance(blocks, list):
      blocks = [blocks]

    vgg = module(blocks, requires_grad=False)
    if cuda != '':
      # self.vgg = cudaify(vgg, cuda)  # Potentially splits model over GPUs
      self.vgg = vgg.cuda()
    else:
      self.vgg = vgg

    self.criterion = _CRITERIONS[criterion]

    if weights is not None:
      assert len(weights) == len(blocks)
      self.weights = weights
    else:
      self.weights = [1.] * len(blocks)

  def forward(self, prediction, target):
    if prediction.shape[1] == 2:  # Handle complex images as inputs
      assert target.shape[1] == 2
      prediction = complex_abs(prediction)
      prediction = torch.cat((prediction, prediction, prediction), dim=1)
      target = complex_abs(target.detach())
      target = torch.cat((target, target, target), dim=1)
    else:
      # Natural images:
      # We assume here that the input is in range (-1, 1)
      prediction = normalize_range(prediction, source_range=(-1., 1.))
      target = normalize_range(target.detach(), source_range=(-1., 1.))

    pred_features = self.vgg(prediction)
    target_features = self.vgg(target)

    loss = 0
    for weight, pred_feature, target_feature in zip(self.weights,
                                                    pred_features,
                                                    target_features):
      loss += weight * self.criterion(pred_feature, target_feature.detach())

    return loss
