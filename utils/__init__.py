
def set_cuda_env(gpu_idx):
  """Sets CUDA_VISIBLE_DEVICES environment variable

  Parameters
  ----------
  gpu_idx : string
    Index of GPU to use, `auto`, or empty string . If `auto`, attempts to
    automatically select a free GPU.

  Returns
  -------
    Value environment variable has been set to

  Raises
  ------
    Exception if auto selecting GPU has been attempted, but failed
  """
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
  return gpu_idx


def set_random_seeds(seed):
  import random
  import numpy as np
  import torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def set_worker_seeds(worker_id):
  import torch
  # Pytorch seed is unique per worker, so we can use it to
  # initialize the other seeds
  set_random_seeds(torch.initial_seed() % (2**32 - 1))


def cpuify(modules_or_tensors):
  if isinstance(modules_or_tensors, dict):
    return {key: cpuify(values) for key, values in modules_or_tensors.items()}
  elif isinstance(modules_or_tensors, (tuple, list)):
    return [cpuify(obj) for obj in modules_or_tensors]

  if modules_or_tensors is not None:
    return modules_or_tensors.cpu()
  else:
    return None


def cudaify(modules_or_tensors, device_ids=None):
  if isinstance(modules_or_tensors, dict):
    return {key: cudaify(values, device_ids)
            for key, values in modules_or_tensors.items()}
  elif isinstance(modules_or_tensors, (tuple, list)):
    return [cudaify(obj, device_ids) for obj in modules_or_tensors]

  if device_ids is not None and not device_ids.isnumeric():
    # Multi-GPU requested: device_ids has the form of '2,3'
    from torch.nn import Module
    from utils.custom_data_parallel import CustomDataParallel
    if isinstance(modules_or_tensors, Module):
      # As we set CUDA_VISIBLE_DEVICES beforehand, device_ids needs to
      # start from zero (i.e. in the form of '0,1')
      device_ids = range(len(device_ids.split(',')))
      return CustomDataParallel(modules_or_tensors,
                                device_ids=device_ids).cuda()
    else:
      return modules_or_tensors.cuda()  # Tensors are sent to default device
  else:
    return modules_or_tensors.cuda()  # Single GPU: send to default device


def make_variables(tensors, volatile):
  from torch.autograd import Variable
  if isinstance(tensors, dict):
    return {key: Variable(tensor, volatile=volatile)
            for key, tensor in tensors.items()}
  elif isinstance(tensors, (tuple, list)):
    return [Variable(tensor, volatile=volatile) for tensor in tensors]
  else:
    return Variable(tensors, volatile=volatile)


def make_fresh_variables(variables, volatile):
  from torch.autograd import Variable
  if isinstance(variables, dict):
    return {key: make_fresh_variables(variable, volatile=volatile)
            for key, variable in variables.items()}
  elif isinstance(variables, (tuple, list)):
    return [Variable(variable.data, volatile=volatile) for variable in variables]
  else:
    return Variable(variables.data, volatile=volatile)


def make_variable_like(tensor, variable):
  from torch import Tensor
  from torch.autograd import Variable
  if isinstance(variable, Tensor):
    return tensor
  requires_grad = variable.requires_grad
  volatile = variable.volatile
  tensor = tensor.type_as(variable.data)
  return Variable(tensor, requires_grad=requires_grad, volatile=volatile)


def import_function_from_path(import_path):
  import importlib

  path_elems = import_path.split('.')
  fn_name = path_elems[-1]

  if len(path_elems) > 1:
    module_path = '.'.join(path_elems[:-1])
    module = importlib.import_module(module_path)
    if hasattr(module, fn_name):
      fn = getattr(module, fn_name)
    else:
      raise ValueError('Could not find {} in module {}'.format(fn_name,
                                                               module_path))
  else:
    if fn_name in globals():
      fn = globals()[fn_name]
    elif fn_name in locals():
      fn = locals()[fn_name]
    else:
      raise ValueError('Could not find {}'.format(fn_name))

  return fn
