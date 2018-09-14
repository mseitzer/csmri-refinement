import inspect
import logging
from itertools import chain

import torch
import torch.nn as nn
from torch.autograd import Variable

from models import construct_model as build_model
from utils.checkpoints import initialize_pretrained_model
from utils.config import Configuration

REQUIRED_PARAMS = [
    'pretrained_model', 'learnable_model'
]

OPTIONAL_PARAMS = [
    'mode', 'input_mode', 'freeze_pretrained_model'
]

KEY_RENAMES = {
    'pretrained_model': 'pretrained_model_conf',
    'learnable_model': 'learnable_model_conf',
}


def construct_model(conf, model_name, **kwargs):
  params = conf.to_param_dict(REQUIRED_PARAMS, OPTIONAL_PARAMS, KEY_RENAMES)
  model_conf = Configuration.from_dict(params['pretrained_model_conf'], conf)
  params['pretrained_model_conf'] = model_conf
  model_conf = Configuration.from_dict(params['learnable_model_conf'], conf)
  params['learnable_model_conf'] = model_conf

  model = RefinementWrapper(**params)
  initialize_pretrained_model(params['pretrained_model_conf'],
                              model.pretrained_model,
                              kwargs['cuda'], conf.file)

  if params.get('freeze_pretrained_model', True):
    # Freeze pretrained model
    for param in model.pretrained_model.parameters():
      param.requires_grad = False

  return model


def _var_without_grad(var):
  return Variable(var.data, requires_grad=False, volatile=var.volatile)


def _scale(tensor):
  """Scale a tensor based on min and max of each example and channel

  Resulting tensor has range (-1, 1).

  Parameters
  ----------
  tensor : torch.Tensor or torch.autograd.Variable
    Tensor to scale of shape BxCxHxW
  Returns
  -------
    Tuple (scaled_tensor, min, max), where min and max are tensors
    containing the values used for normalizing the tensor
  """
  b, c, h, w = tensor.shape
  out = tensor.view(b, c, h * w)
  minimum, _ = out.min(dim=2, keepdim=True)
  out = out - minimum
  maximum, _ = out.max(dim=2, keepdim=True)
  out = out / maximum  # out has range (0, 1)
  out = out * 2 - 1  # out has range (-1, 1)

  return out.view(b, c, h, w), minimum, maximum


def _unscale(tensor, minimum, maximum):
  """Rescale a scaled tensor from range (-1, 1) back to its original range

  Parameters
  ----------
  tensor : torch.Tensor or torch.autograd.Variable
    Tensor to rescale with range (-1, 1) of shape BxCxHxW
  mimimum : torch.Tensor
    Minimum used to scale the tensor of shape 1xB*CxHxW
  maximum : torch.Tensor
    Maximum used to scale the tensor of shape 1xB*CxHxW
  """
  b, c, h, w = tensor.shape
  out = tensor.view(b, c, h * w)
  out = (out + 1) / 2  # out has range (0, 1)
  out = out * maximum + minimum  # out has original range
  return out.view(b, c, h, w)


class RefinementWrapper(nn.Module):
  """Model that uses a pretrained path and a learnable path and adds the two"""
  def __init__(self, pretrained_model_conf, learnable_model_conf,
               mode='add', input_mode='input', mse_path_model_conf=None,
               freeze_pretrained_model=True, disable_strict_loading=False):
    super(RefinementWrapper, self).__init__()
    self.mode = mode
    self.freeze_pretrained_model = freeze_pretrained_model
    self.pretrained_model = build_model(pretrained_model_conf,
                                        pretrained_model_conf.name)
    self.learnable_model = build_model(learnable_model_conf,
                                       learnable_model_conf.name)

    if mode == 'add':
      self._refine_op = self._refinement_add
    elif mode == 'real-penalty-add':
      self.scale = nn.Parameter(torch.zeros(1))
      self._refine_op = self._refinement_real_penalty_add
    else:
      raise ValueError('Unknown mode {}'.format(mode))

    if input_mode == 'input':
      self._learnable_model_input_fn = lambda inp, out: inp
    elif input_mode == 'output':
      self._learnable_model_input_fn = lambda inp, out: out
    elif input_mode == 'concat':
      self._learnable_model_input_fn = lambda inp, out: torch.cat((inp, out),
                                                                  dim=1)
    else:
      raise ValueError('Unknown input mode {}'.format(mode))

    # As models can have different arguments on their forward function,
    # we need to dynamically select a forward function which fits the
    # signature of the pretrained model. For now we assume that the learnable
    # model always has one input, and that only the pretrained model can have
    # different signatures.
    forward_replacements = [
        self._forward_vanilla,
        self._forward_reconstruction
    ]

    signature_pretrained = inspect.signature(self.pretrained_model.forward)
    params_pretrained = signature_pretrained.parameters
    for forward_fn in forward_replacements:
      if params_pretrained == inspect.signature(forward_fn).parameters:
        self.forward = forward_fn
        break
    else:
      raise RuntimeError(('Could not find fitting forward '
                          'function with params {}').format(params_pretrained))

  def parameters(self):
    """Overwrite parameters to exclude frozen parameters of pretrained model"""
    if self.mode == 'real-penalty-add-mse-scale':
      return {
          'adversarial_path': chain(self.learnable_model.parameters(),
                                    [self.scale]),
          'mse_path': self.mse_path_model.parameters()
      }
    elif not self.freeze_pretrained_model:
      return {
          'adversarial_path': chain(self.learnable_model.parameters(),
                                    [self.scale]),
          'pretrained_path': self.pretrained_model.parameters()
      }
    else:
      params = super(RefinementWrapper, self).parameters()
      return filter(lambda p: p.requires_grad, params)

  def _refinement_add(self, inp, out_pretrained):
    learn_model_input = self._learnable_model_input_fn(inp, out_pretrained)
    out_learnable = self.learnable_model(learn_model_input)
    return out_pretrained + out_learnable

  def _refinement_real_penalty_add(self, inp, out_pretrained):
    pretrained_real = out_pretrained[:, 0].unsqueeze(1).contiguous()
    pretrained_imag = out_pretrained[:, 1].unsqueeze(1).contiguous()

    # Transform to range (-1, 1)
    pretrained_real_scaled, minimum, maximum = _scale(pretrained_real)

    learn_model_input = self._learnable_model_input_fn(inp, out_pretrained)
    out_learnable = self.learnable_model(learn_model_input)

    out_learnable_scaled = self.scale * out_learnable
    refined = pretrained_real_scaled + out_learnable_scaled

    # Transform using original range of image
    # After adding, the range is outside of (-1, 1), so the range of the
    # backtransformed image will not match the range of the original image.
    # This is okay though, as we want to allow the refinement to modify
    # the range of the image
    out_real = _unscale(refined, minimum, maximum)

    return {
        'pred': torch.cat((out_real, pretrained_imag), dim=1),
        'pretrained': out_pretrained,
        'prescaled_refinement': out_learnable,
        'scaled_refinement': out_learnable_scaled,
    }

  def _forward_vanilla(self, inp):
    """Normal one input forward function"""
    if self.freeze_pretrained_model:
      inp = _var_without_grad(inp)

    out_pretrained = self.pretrained_model(inp)

    if self.freeze_pretrained_model:
      out_pretrained = out_pretrained.detach()

    return self._refine_op(inp, out_pretrained)

  def _forward_reconstruction(self, inp, kspace, mask):
    """Forward function when the pretrained model needs kspace information"""
    if self.freeze_pretrained_model:
      inp = _var_without_grad(inp)
      kspace = _var_without_grad(kspace)
      mask = _var_without_grad(mask)

    out_pretrained = self.pretrained_model(inp, kspace, mask)

    if self.freeze_pretrained_model:
      out_pretrained = out_pretrained.detach()

    return self._refine_op(inp, out_pretrained)

