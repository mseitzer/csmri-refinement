import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel._functions import Gather


def gather(outputs, target_device, dim=0):
  r"""
  Gathers variables from different GPUs on a specified device
  (-1 means the CPU).
  """
  def gather_map(outputs):
    out = outputs[0]
    if isinstance(out, Variable):
      return Gather.apply(target_device, dim, *outputs)
    if out is None:
      return None
    if isinstance(out, dict):
      # Patch to support dictionaries
      value_iter = (item.values() for item in outputs)
      return dict(zip(out, map(gather_map, zip(*value_iter))))
    return type(out)(map(gather_map, zip(*outputs)))

  return gather_map(outputs)


class CustomDataParallel(nn.DataParallel):
  """DataParallel with fix for returning dictionaries"""
  def __init__(self, module, device_ids=None, output_device=None, dim=0):
    super(CustomDataParallel, self).__init__(module,
                                             device_ids,
                                             output_device,
                                             dim)

  def gather(self, outputs, output_device):
    return gather(outputs, output_device, dim=self.dim)
