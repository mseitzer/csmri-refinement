import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import calculate_gain

from models.utils import (get_activation_fn, get_activation_fn_name,
                          get_same_padding_layer, get_normalization_layer,
                          need_bias)
from models.weight_inits import initialize_weights


REQUIRED_PARAMS = [
    'num_inputs', 'num_filters_per_layer', 'strides'
]

OPTIONAL_PARAMS = [
    'kernel_sizes', 'fc_layers', 'spatial_shape', 'act_fn',
    'relu_leakiness', 'use_norm_layers', 'norm_layer', 'use_weightnorm',
    'padding', 'final_conv_kernel_size', 'final_average_pooling',
    'use_biases', 'compute_features',
    'dropout_after', 'dropout_prob'
]


def construct_model(conf, model_name, **kwargs):
  if model_name == 'CNNDiscriminator':
    params = conf.to_param_dict(REQUIRED_PARAMS, OPTIONAL_PARAMS)
    model = CNNDiscriminator(**params)
    initialize_weights(model, conf.get_attr('weight_init', default={}))
  else:
    raise ValueError('Unknown discriminator {}'.format(model_name))
  return model


def _get_gain(act_fn):
  gain = calculate_gain('linear')
  if act_fn is not None:
    act_name = get_activation_fn_name(act_fn)
    if act_name == 'relu':
      gain = calculate_gain('relu')
    elif act_name == 'lrelu':
      gain = calculate_gain('leaky_relu', act_fn.negative_slope)
    elif act_name == 'prelu':
      gain = calculate_gain('leaky_relu', act_fn.weight[0].data[0])

  return gain


class CNNDiscriminator(nn.Module):
  """CNN based discriminator network"""
  DEFAULT_RELU_LEAKINESS = 0.2

  def __init__(self, num_inputs, num_filters_per_layer, strides,
               kernel_sizes=None, fc_layers=[], spatial_shape=None,
               act_fn='lrelu', relu_leakiness=DEFAULT_RELU_LEAKINESS,
               use_norm_layers=True, norm_layer='batch', use_weightnorm=False,
               padding='zero', final_conv_kernel_size=1, use_biases=True,
               final_average_pooling=False,
               compute_features=False, dropout_after=[], dropout_prob=0.5):
    """Construct model

    Parameters
    ----------
    num_inputs: int
      Number of input channels
    num_filters_per_layer : list or tuple
      Number of filters the discriminator uses in each layer
    kernel_sizes : list
      Shape of filters in each layer. Defaults to 3
    strides : list
      Strides of filters in each layer.
    fc_layers : list
      Number of channels of fully connected layers after convolutional layers.
      If no fully connected layers are selected, the convolutional features
      maps will be reduced to one dimension with a 1x1 convolution, and the
      output is a probability map (corresponds to a PatchGAN)
    spatial_shape : tuple
      Spatial shape of input in the form of (height, width). Required if
      using fully connected layers
    act_fn : string
      Activation function to use. Either `relu`, `prelu`, or `lrelu` (default)
    relu_leakiness : float
      If using lrelu, leakiness of the relus, if using prelu, initial value
      for prelu parameters
    use_norm_layers : bool or string
      If true, use normalization layers. If `not-first`, skip the normalization
      after the first convolutional layer
    norm_layer : string
      Normalization layer to use. `batch` for batch normalization or `instance`
      for instance normalization
    use_weightnorm : bool
      If true, applies weight norm to all layers
    padding : string
      Type of padding to use. Either `zero`, `reflection`, or `replication`
    final_conv_kernel_size : int
      Shape of filter of final convolution, if using no fc layers.
      Defaults to 1
    final_average_pooling : bool
      If true, reduce spatial dimensions to a scalar using an average pooling
      layer. Only used if no fully connected layers were requested
    use_biases : bool
      If false, deactivate bias on all layers
    compute_features : bool
      If true, return the feature maps from forward pass in the key `features`
    dropout_after : list
      Indices of convolutional layers after which to insert layerwise dropout
    dropout_prob : float
      Probability to drop entries in dropout layers
    """
    super(CNNDiscriminator, self).__init__()
    if len(fc_layers) > 0:
      assert spatial_shape is not None, \
          'Need input spatial shape if using fully connected layers'

    if kernel_sizes is None:
      kernel_sizes = 3
    if isinstance(kernel_sizes, int):
      kernel_sizes = [kernel_sizes] * len(num_filters_per_layer)

    assert len(num_filters_per_layer) == len(strides)
    assert len(num_filters_per_layer) == len(kernel_sizes)

    self.compute_features = compute_features
    self.feature_layers = set()

    in_channels = num_inputs

    layer_idx = 0
    layers = []
    for num_filters, kernel_size, stride in zip(num_filters_per_layer,
                                                kernel_sizes,
                                                strides):
      use_bias = use_biases and need_bias(use_norm_layers, norm_layer)
      layers += (get_same_padding_layer(kernel_size=kernel_size, stride=stride,
                                        mode=padding),
                 nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size,
                           stride=stride,
                           bias=use_bias))
      if use_norm_layers != 'not-first' and use_norm_layers:
        layers.append(get_normalization_layer(norm_layer, num_filters))
      elif use_norm_layers == 'not-first':
        use_norm_layers = True
      layers.append(get_activation_fn(act_fn, relu_leakiness, num_filters))

      if compute_features:
        self.feature_layers.add(layers[-1])

      if layer_idx in dropout_after:
        layers.append(nn.Dropout2d(p=dropout_prob, inplace=True))

      in_channels = num_filters
      layer_idx += 1

    self.convs = nn.Sequential(*layers)

    if len(fc_layers) > 0:
      input_dims = self._infer_shape(self.convs, num_inputs, spatial_shape)

      layers = []
      for num_features in fc_layers[:-1]:
        layers += (nn.Linear(input_dims, num_features, bias=use_biases),
                   get_activation_fn(act_fn, relu_leakiness, num_features))
        input_dims = num_features

      layers.append(nn.Linear(input_dims, fc_layers[-1]))

      self.fcs = nn.Sequential(*layers)
      self.final_conv = None
    else:
      self.fcs = None
      final_conv = [nn.Conv2d(in_channels, out_channels=1,
                              kernel_size=final_conv_kernel_size, stride=1,
                              bias=use_biases)]
      if final_average_pooling:
        final_conv.append(nn.AdaptiveAvgPool2d((1, 1)))

      self.final_conv = nn.Sequential(*final_conv)

  @staticmethod
  def _infer_shape(model, num_inputs, spatial_shape):
    """Infer shape by doing a forward pass"""
    inp = Variable(torch.ones(1, num_inputs,
                              spatial_shape[0], spatial_shape[1]),
                   volatile=True)
    outp = model(inp)
    return outp.view(1, -1).shape[1]

  def weight_init_params(self, user_weight_init=None):
    init_params = {
        'conv_weight': ('normal', 0.0, 0.02),
        'linear_weight': ('normal', 0.0, 0.02),
        'batchnorm_weight': ('normal', 1.0, 0.02)
    }

    if user_weight_init is not None and 'final_layer_bias' in user_weight_init:
      final_layer_bias_init = user_weight_init['final_layer_bias']
      if self.fcs is not None:
        layer = self.fcs[-1]
        assert isinstance(layer, nn.Linear)
      else:
        if isinstance(self.final_conv[-1], nn.Conv2d):
          layer = self.final_conv[-1]
        else:
          layer = self.final_conv[-2]
        assert isinstance(layer, nn.Conv2d)
      init_params[layer] = {'bias': final_layer_bias_init}

    return init_params

  def forward(self, inp):
    if self.compute_features:
      features = []
      x = inp

      for layer in self.convs:
        x = layer(x)
        if layer in self.feature_layers:
          features.append(x)

      if self.fcs is not None:
        x = x.view(x.shape[0], -1)
        for fc in self.fcs:
          x = fc(x)
          features.append(x)
      else:
        x = self.final_conv(x)
        features.append(x)

      return {
          'prob': F.sigmoid(x),
          'logits': x,
          'features': features
      }
    else:
      x = self.convs(inp)

      if self.fcs is not None:
        x = x.view(x.shape[0], -1)
        x = self.fcs(x)
      else:
        x = self.final_conv(x)

      return {
          'prob': F.sigmoid(x),
          'logits': x
      }
