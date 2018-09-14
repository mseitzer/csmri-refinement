import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from metrics.segmentation_metrics import compute_dice
from models import construct_model
from utils.checkpoints import initialize_pretrained_model


class SegmentationScore(torch.nn.Module):
  """Metric that measures dice score using a pretrained segmentation network"""
  def __init__(self, model_conf, conf_path, cuda, class_idx,
               save_segmentations_path=None,
               skip_empty_images=False):
    super(SegmentationScore, self).__init__()
    self.cuda = cuda
    self.model = construct_model(model_conf, model_conf.name)
    self.class_idxs = class_idx
    if not isinstance(self.class_idxs, list):
      self.class_idxs = [self.class_idxs]
    self.skip_empty_images = skip_empty_images

    initialize_pretrained_model(model_conf, self.model, cuda, conf_path)

    if cuda != '':
      self.model = self.model.cuda()

    self.model.eval()

    self.save_segmentations_path = save_segmentations_path
    if save_segmentations_path is not None:
      parent_dir = os.path.dirname(save_segmentations_path)
      assert os.path.isdir(parent_dir), \
          'Did not find path {}'.format(parent_dir)
      if not os.path.isdir(save_segmentations_path):
        os.mkdir(save_segmentations_path)
      self.num_saved_segmns = 0

  def _save_segmentations(self, segmentations):
    for segmentation in segmentations:
      self.num_saved_segmns += 1
      path = os.path.join(self.save_segmentations_path,
                          '{:04d}_segm.npy'.format(self.num_saved_segmns))
      np.save(path, segmentation.data.cpu().numpy().astype(np.uint8))

  def forward(self, prediction, target):
    assert isinstance(prediction, Variable)
    assert isinstance(target, Variable)

    prediction = Variable(prediction.data, volatile=True)
    target = Variable(target.data, volatile=True)

    if self.skip_empty_images:
      # Skip images which do not contain at least one target class
      skip = True
      for class_idx in self.class_idxs:
        if (target.data == class_idx).sum() != 0:
          skip = False
          break
      if skip:
        return None

    segmentation = self.model(prediction)

    probs = F.softmax(segmentation, dim=1)
    _, predicted_classes = probs.max(dim=1)

    if self.save_segmentations_path is not None:
      self._save_segmentations(predicted_classes)

    dices = []
    for class_idx in self.class_idxs:
      dices.append(compute_dice(predicted_classes, target, class_idx,
                                absent_value=1.0))

    return sum(dices) / len(dices)
