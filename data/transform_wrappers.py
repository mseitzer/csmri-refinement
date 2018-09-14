"""Wrappers building transformation from configuration"""
import logging

from PIL import Image

from utils.config import Configuration


def _build_param_dict(conf, required_params, optional_params=[],
                      key_renames={}, kwargs={}):
  # Filter out params which are passed in kwargs
  required_params = [p for p in required_params if p not in kwargs]
  param_dict = conf.to_param_dict(required_params,
                                  optional_params.copy(),
                                  key_renames)
  param_dict.update(kwargs)
  return param_dict


def get_rec_transform(conf, mode, **kwargs):
  assert mode in ('train', 'test', 'inference')
  required_params = ['undersampling', 'image_size']
  key_renames = {
      'undersampling': 'cs_params'
  }

  if mode == 'train':
    from data.reconstruction.rec_transforms import train_transform
    param_dict = _build_param_dict(conf,
                                   required_params=required_params,
                                   optional_params={
                                       'downscale': 1,
                                       'augmentation': None
                                   },
                                   key_renames=key_renames,
                                   kwargs=kwargs)
    transform = train_transform(**param_dict)
  else:
    from data.reconstruction.rec_transforms import test_transform
    param_dict = _build_param_dict(conf,
                                   required_params=required_params,
                                   optional_params={'downscale': 1},
                                   key_renames=key_renames,
                                   kwargs=kwargs)
    transform = test_transform(**param_dict)

  return transform


def get_rec_seg_transform(conf, mode, **kwargs):
  assert mode in ('train', 'test', 'inference')
  required_params = ['undersampling', 'image_size']
  key_renames = {
      'undersampling': 'cs_params'
  }

  if mode == 'train':
    from data.reconstruction.rec_seg_transforms import train_transform
    param_dict = _build_param_dict(conf,
                                   required_params=required_params,
                                   optional_params={
                                       'downscale': 1,
                                       'augmentation': None,
                                   },
                                   key_renames=key_renames,
                                   kwargs=kwargs)
    transform = train_transform(**param_dict)
  else:
    from data.reconstruction.rec_seg_transforms import test_transform
    param_dict = _build_param_dict(conf,
                                   required_params=required_params,
                                   optional_params={'downscale': 1},
                                   key_renames=key_renames,
                                   kwargs=kwargs)
    transform = test_transform(**param_dict)

  return transform


def get_rec_output_transform(conf, mode, **kwargs):
  from data.reconstruction.rec_transforms import output_transform
  transform = output_transform()
  return transform


def get_seg_output_transform(conf, mode, **kwargs):
  from data.reconstruction.seg_transforms import output_transform
  transform = output_transform()
  return transform


def get_output_transform(conf, application, mode, **kwargs):
  applications = {
      'reconstruction': get_rec_output_transform,
      'segmentation': get_seg_output_transform,
      'none': None
  }

  assert application in applications
  if applications[application] is None:
    logging.debug(('Unknown application {} for output transform. Using no '
                   'output transform').format(application))
    return None
  return applications[application](conf, mode, **kwargs)


def get_rec_input_batch_transform(conf, mode, **kwargs):
  assert mode in ('train', 'test')
  return None


def get_input_batch_transform(conf, application, mode, **kwargs):
  applications = {
      'reconstruction': get_rec_input_batch_transform,
      'segmentation': None,
      'none': None,
  }

  assert application in applications
  if applications[application] is None:
    logging.debug(('Unknown application {} for input batch transform. Using '
                   'no input batch transform').format(application))
    return None
  return applications[application](conf, mode, **kwargs)
