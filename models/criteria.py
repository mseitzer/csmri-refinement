import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.config import Configuration


def _get_adv_criterion(conf, loss_name, cuda, target_key, loss_type):
  from models.adversarial_loss import get_adversarial_loss
  # Adversarial losses don't follow the (prediction, target) structure,
  # so we don't wrap the criterion with the CriterionWrapper.
  return get_adversarial_loss(conf, loss_name, cuda, loss_type)


def _get_vgg_criterion(conf, loss_name, cuda, target_key):
  from models.vgg_loss import VGGLoss

  if conf.has_attr('vgg_loss'):
    blocks = conf.vgg_loss.get('blocks', -1)
    criterion = conf.vgg_loss.get('criterion', 'MSE')
    weights = conf.vgg_loss.get('weights')
  else:
    blocks = -1
    criterion = 'MSE'
    weights = None

  vgg_loss = VGGLoss(loss_name, cuda, blocks, criterion, weights)
  return CriterionWrapper(vgg_loss, target_key)


def _get_feature_penalty_criterion(conf, loss_name, cuda, target_key):
  assert conf.has_attr('feature_penalty'), \
      ('Feature penalty loss needs additional config under key '
       '"feature_penalty"')
  assert 'input_key' in conf.feature_penalty, \
      ('Feature penalty loss needs input key specifying which model output to'
       ' apply the penalty to as additional config under key "input_key"')
  input_key = conf.feature_penalty['input_key']

  criterion = conf.feature_penalty.get('criterion', 'MSE')
  assert criterion in _CRITERIA, \
      'Unknown criterion {} for feature penalty loss'.format(criterion)

  criterion_constructor = _CRITERIA[criterion]
  return CriterionWrapperWithScalarTarget(criterion_constructor(), cuda,
                                          scalar_target=0.0,
                                          input_key=input_key)


_CRITERIA = {
    'MSE': nn.MSELoss,
    'L1': nn.L1Loss,
    'SmoothL1Loss': nn.SmoothL1Loss,
    'CrossEntropy': nn.CrossEntropyLoss,
    'NLLLoss': nn.NLLLoss2d,
    'GAN': _get_adv_criterion,
    'LSGAN': _get_adv_criterion,
    'WGAN': _get_adv_criterion,
    'FeatureMatching': _get_adv_criterion,
    'VGG19': _get_vgg_criterion,
    'FeaturePenalty': _get_feature_penalty_criterion,
    # Legacy
    'gan': _get_adv_criterion,
    'lsgan': _get_adv_criterion,
    'feature-matching': _get_adv_criterion,
}


class CriterionWrapper(nn.Module):
  """Class wrapping criterions to select input and targets"""
  def __init__(self, criterion, target_key='target', input_key='pred'):
    super(CriterionWrapper, self).__init__()
    self.criterion = criterion
    self.target_key = target_key
    self.input_key = input_key

  def forward(self, out_gen, batch):
    if isinstance(out_gen, dict):
      prediction = out_gen[self.input_key]
    else:
      prediction = out_gen

    return self.criterion(prediction, batch[self.target_key])


class CriterionWrapperWithScalarTarget(CriterionWrapper):
  def __init__(self, criterion, cuda, scalar_target, input_key='pred'):
    super(CriterionWrapperWithScalarTarget, self).__init__(criterion,
                                                           input_key=input_key)
    self.scalar_target = scalar_target
    self.target = None  # Holds the target tensor

    if cuda != '':
      self.tensor_fn = lambda *args: torch.FloatTensor(*args).cuda()
    else:
      self.tensor_fn = lambda *args: torch.FloatTensor(*args)

  def forward(self, out_gen, batch):
    if isinstance(out_gen, dict):
      prediction = out_gen[self.input_key]
    else:
      prediction = out_gen

    target_shape = prediction.shape
    if self.target is None or self.target.shape != target_shape:
      tensor = self.tensor_fn(target_shape).fill_(self.scalar_target)
      self.target = Variable(tensor, requires_grad=False)

    return self.criterion(prediction, self.target)


def get_criterion(conf, loss_name, cuda,
                  target_key=None, input_key=None, **kwargs):
  assert loss_name in _CRITERIA, 'Unknown loss {}'.format(loss_name)
  criterion_constructor = _CRITERIA[loss_name]

  if input_key is None:
    input_key = 'pred'
  if target_key is None:
    target_key = conf.get_attr('loss_target_keys', default={}).get(loss_name,
                                                                   'target')

  if isinstance(criterion_constructor, type):
    # Class: probably directly pytorch criterion
    return CriterionWrapper(criterion_constructor(), target_key, input_key)
  else:
    # Function: pass additional information
    return criterion_constructor(conf, loss_name, cuda, target_key, **kwargs)
