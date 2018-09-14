import numpy as np
import torch.nn as nn


def get_model_parameter_str(model):
  """Get a summary string with model parameters

  Adapted from https://stackoverflow.com/a/45528544
  """
  def is_leaf(module):
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
      return False
    elif any((isinstance(module, (nn.ModuleList, nn.Sequential))
              for module in module._modules.values())):
      return False

    return all((is_leaf(m) for m in module._modules.values()))

  from torch.nn.modules.module import _addindent
  tmpstr = model.__class__.__name__ + ' (\n'

  total_params = 0
  for key, module in model._modules.items():
    if module is None:
      continue

    if is_leaf(module):
      modstr = module.__repr__()
      leaf = True
    else:
      modstr = get_model_parameter_str(module)
      leaf = False

    modstr = _addindent(modstr, 2)

    params = sum([np.prod(p.size()) for p in module.parameters()])
    total_params += params

    tmpstr += '  (' + key + '): ' + modstr
    if leaf and params > 0:
      tmpstr += ', parameters={}'.format(params)
    tmpstr += '\n'

  tmpstr = tmpstr + ')'
  if total_params > 0:
    tmpstr += ', parameters={}'.format(total_params)

  return tmpstr


def print_model_parameters(runner):
  models_by_name = runner.get_named_models()
  for name, model in models_by_name.items():
    print(name)
    print(get_model_parameter_str(model))

