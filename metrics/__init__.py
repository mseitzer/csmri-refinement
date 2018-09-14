from functools import partial

from data.transform_wrappers import get_output_transform
from metrics.metric import MaxMetric, MinMetric
from utils import import_function_from_path
from utils.config import Configuration


class MetricFunction(object):
  def __init__(self, metric_fn, metric_type, transform=None,
               pred_key='pred', target_key='target'):
    """Metric class which wraps metric computing functions

    Parameters
    ----------
    metric_fn : function
      Function which takes prediction and target and returns the metric value
    metric_type : int
      Either `MIN_METRIC`, which indicates that for this metric, lower values
      are better, or `MAX_METRIC`, which indicates that for this metric,
      higher values are better
    transform : function pred, target -> pred, target
      Optional function transforming prediction and target to the format the
      metric expects
    pred_key : str
      Selects the entry to pass to the metric function if passing a dictionary
      to this function for the prediction argument
    target_key : str
      Selects the entry to pass to the metric function if passing a dictionary
      to this function for the target argument
    """
    self.metric_fn = metric_fn
    self.metric_type = metric_type
    self.transform = transform
    self.pred_key = pred_key
    self.target_key = target_key

  def __call__(self, prediction, target, transform=True):
    """Computes value of metric between prediction and target

    Parameters
    ----------
    prediction : torch.autograd.Variable
      Predicted value
    target : torch.autograd.Variable
      Target value
    transform : bool
      If true, apply transformation function on the input if any
    """
    if isinstance(prediction, dict):
      prediction = prediction[self.pred_key]
    if isinstance(target, dict):
      target = target[self.target_key]

    if transform and self.transform is not None:
      prediction, target = self.transform(prediction, target)

    if prediction.dim() == 3 or prediction.dim() == 4:
      if target is not None:
        # If we get a batch, compute metrics for every item of the batch
        # independently
        values = (self.metric_fn(p.unsqueeze(0), t.unsqueeze(0))
                  for p, t in zip(prediction, target))
      else:
        values = (self.metric_fn(p.unsqueeze(0), None)
                  for p in prediction)
    else:
      values = [self.metric_fn(prediction, target)]

    values = (v for v in values if v is not None)

    return self.metric_type(values)


def _get_segmentation_score_metric(conf, metric_name, cuda):
  from metrics.segmentation_score import SegmentationScore
  assert conf.has_attr('segmentation_score_metric'), \
      ('Segmentation score metric needs additional config '
       'under key "segmentation_score_metric"')

  metric_conf = conf.segmentation_score_metric
  model_conf = Configuration.from_dict(metric_conf['model'])
  dice_score_class = metric_conf.get('class')
  save_segmentations_path = metric_conf.get('save_segmentations_path')
  skip_empty_images = metric_conf.get('skip_empty_images', False)

  return SegmentationScore(model_conf, conf.file, cuda, dice_score_class,
                           save_segmentations_path, skip_empty_images)


def _get_average_dice_metric(conf, metric_name, cuda):
  from metrics.segmentation_metrics import compute_average_dice
  assert conf.has_attr('dice_metric'), \
      ('Dice metric needs additional config '
       'under key "dice_metric"')

  metric_conf = conf.dice_metric
  assert 'num_classes' in metric_conf, \
      'Dice metric needs number of classes under key "num_classes"'

  exclude_bg = metric_conf.get('exclude_background', False)

  return partial(compute_average_dice,
                 num_classes=metric_conf['num_classes'],
                 excluded_class=0 if exclude_bg else -1)


def _get_disc_accuracy_metric(conf, metric_name, cuda):
  from metrics.scalar_metrics import disc_accuracy
  if metric_name == 'accuracy_fake' or metric_name == 'binary_accuracy':
    fake_accuracy = True
    real_accuracy = False
  elif metric_name == 'accuracy_real':
    fake_accuracy = False
    real_accuracy = True
  elif metric_name == 'accuracy':
    fake_accuracy = True
    real_accuracy = True
  else:
    raise ValueError('Unsupported metric {}'.format(metric_name))

  return partial(disc_accuracy,
                 fake_accuracy=fake_accuracy,
                 real_accuracy=real_accuracy,
                 cuda=cuda)


def _get_generic_metric_fn(import_path, **kwargs):
  fn = import_function_from_path(import_path)

  if len(kwargs) > 0:
    return partial(fn, **kwargs)
  else:
    return fn


_METRICS = {
    'psnr': ('metrics.image_metrics.compute_psnr', MaxMetric),
    'ssim': ('metrics.image_metrics.compute_ssim', MaxMetric),
    'hfen': ('metrics.image_metrics.compute_hfen', MinMetric),
    'mutual_information': ('metrics.image_metrics.compute_mutual_information',
                           MaxMetric),
    # Have to map binary_accuracy to disc_accuracy for backwards compatibility
    'binary_accuracy': (_get_disc_accuracy_metric, MaxMetric),
    'accuracy': (_get_disc_accuracy_metric, MaxMetric),
    'accuracy_fake': (_get_disc_accuracy_metric, MaxMetric),
    'accuracy_real': (_get_disc_accuracy_metric, MaxMetric),
    'dice': (_get_average_dice_metric, MaxMetric),
    'dice_class_0': ('metrics.segmentation_metrics.compute_dice', MaxMetric,
                     {'class_idx': 0}),
    'dice_class_1': ('metrics.segmentation_metrics.compute_dice', MaxMetric,
                     {'class_idx': 1}),
    'dice_class_2': ('metrics.segmentation_metrics.compute_dice', MaxMetric,
                     {'class_idx': 2}),
    'dice_class_3': ('metrics.segmentation_metrics.compute_dice', MaxMetric,
                     {'class_idx': 3}),
    'segmentation_score': (_get_segmentation_score_metric, MaxMetric)
}


def get_metric_fn(conf, metric_name, cuda, mode,
                  pred_key='pred', target_key='target'):
  assert mode in ('train', 'test')
  assert metric_name in _METRICS, 'Unknown metric {}'.format(metric_name)

  metric_info = _METRICS[metric_name]
  metric_type = metric_info[1]

  if isinstance(metric_info[0], str):
    # Generic metric function
    metric_path = metric_info[0]
    if len(metric_info) <= 2:
      metric_fn = _get_generic_metric_fn(metric_path)
    else:
      metric_fn = _get_generic_metric_fn(metric_path, **metric_info[2])
  else:
    # Metric requires specialized handling
    metric_constructor = metric_info[0]
    metric_fn = metric_constructor(conf, metric_name, cuda)

  metric_conf = conf.get_attr('{}_metric'.format(metric_name), default={})
  if 'pred_key' in metric_conf:
    pred_key = metric_conf['pred_key']
  if 'target_key' in metric_conf:
    target_key = metric_conf['target_key']

  if 'transform' in metric_conf:
    transform = metric_conf['transform']
    if transform == 'none':
      transform = None
    else:
      transform = get_output_transform(conf, transform, mode)
  else:
    transform = get_output_transform(conf, conf.application, mode)

  return MetricFunction(metric_fn, metric_type, transform,
                        pred_key, target_key)


def get_loss_metric(value):
  return MinMetric(value)


def accumulate_metric(dictionary, metric_name, metric):
  if metric_name in dictionary:
    dictionary[metric_name].accumulate(metric)
  else:
    dictionary[metric_name] = metric
