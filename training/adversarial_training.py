from enum import Enum, auto

import torch
from torch.nn.functional import softmax

from utils import make_variable_like
from utils.image_pool import ImagePool
from utils.tensor_transforms import (scale_batch_per_example,
                                     normalize_batch_per_example,
                                     convert_to_one_hot)


DEFAULT_INPUT_METHOD = 'simple'


class CondInputSource(Enum):
  INPUT = auto()
  OUT_GEN = auto()


def _build_input_fn(method, normalize=False, image_pool=None,
                    cond_input_source=CondInputSource.INPUT,
                    cond_input_gen_key=None,
                    strip_bg_class=False,
                    scale_input_to_zero_one=False,
                    pool_label_swapping=False):
  def simple(prediction, cond_inp):
    if normalize:
      prediction = normalize_batch_per_example(prediction)

    return prediction

  def simple_magnitude(prediction, cond_inp):
    """Returns magnitude image of complex input"""
    from utils.tensor_transforms import complex_abs
    prediction = complex_abs(prediction)
    if normalize:
      prediction = normalize_batch_per_example(prediction)

    return prediction

  def pool_wrapper(prediction, inp, out_gen, is_real_input, detach=False):
    disc_input = input_wrapper(prediction, inp, out_gen, is_real_input, detach)
    # If gradients on the generator are required (detach=False), we can not
    # query the image pool, as pool outputs are automatically detached.
    if detach:
        # We can only query the pool for fake images, or if we employ pool label
        # swapping, i.e. there are real and fake images in the pool and they are
        # used with real and fake discriminator targets
        if not is_real_input or pool_label_swapping:
          return image_pool.query(disc_input)

    return disc_input

  def input_wrapper(prediction_or_target, inp, out_gen,
                    is_real_input, detach=False):
    if isinstance(prediction_or_target, dict):
      prediction = prediction_or_target['pred']
    else:
      prediction = prediction_or_target

    if strip_bg_class:
      prediction = prediction[:, 1:]

    if cond_input_source is CondInputSource.INPUT:
      conditional_input = inp
    elif cond_input_source is CondInputSource.OUT_GEN:
      conditional_input = out_gen[cond_input_gen_key]

    if scale_input_to_zero_one:
      conditional_input = scale_batch_per_example(conditional_input)

    if detach:
      prediction = prediction.detach()
      conditional_input = conditional_input.detach()

    return input_fn(prediction, conditional_input)

  methods = {
      'simple': simple,
      'simple-magnitude': simple_magnitude,
  }

  assert method in methods, \
      'Unknown discriminator input method {}'.format(method)

  input_fn = methods[method]

  if image_pool is not None:
    return pool_wrapper
  else:
    return input_wrapper


def get_discriminator_input_fn(conf, disc_conf, no_pool=False):
  if disc_conf.get_attr('use_image_pool', default=False) and not no_pool:
    pool_size = disc_conf.get_attr('image_pool_size',
                                   default=5 * conf.batch_size)
    sample_prob = disc_conf.get_attr('image_pool_sample_prob', default=0.5)
    image_pool = ImagePool(pool_size, sample_prob)
  else:
    image_pool = None

  pool_label_swapping = disc_conf.get_attr('image_pool_label_swapping',
                                           default=False)

  input_method = disc_conf.get_attr('input_method',
                                    default=DEFAULT_INPUT_METHOD)
  normalize_input = disc_conf.get_attr('normalize_input', default=False)
  scale_input = disc_conf.get_attr('scale_input_zero_one', default=False)

  strip_bg_class = disc_conf.get_attr('strip_bg_class', default=False)

  cond_input_src = disc_conf.get_attr('conditional_input_source',
                                      default='input')
  if cond_input_src == 'input':
    cond_input_src = CondInputSource.INPUT
  elif cond_input_src == 'generator':
    cond_input_src = CondInputSource.OUT_GEN
  else:
    raise ValueError(('Unknown conditional '
                     'input source {}').format(cond_input_src))

  cond_input_gen_key = disc_conf.get_attr('conditional_input_generator_key')

  disc_input_fn = _build_input_fn(input_method,
                                  normalize_input,
                                  image_pool,
                                  cond_input_src,
                                  cond_input_gen_key,
                                  strip_bg_class,
                                  scale_input,
                                  pool_label_swapping)

  return disc_input_fn
