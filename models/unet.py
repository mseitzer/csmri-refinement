import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_same_padding_layer
from models.weight_inits import initialize_weights

REQUIRED_PARAMS = [
    'num_inputs', 'num_outputs', 'num_layers_per_scale',
    'encode_filters', 'decode_filters', 'output_activation'
]

OPTIONAL_PARAMS = [
    'kernel_size', 'transposed_kernel_size', 'relu_leakiness', 'use_bn',
    'upsampling_mode', 'padding', 'encoder_features',
    'use_refinement', 'decoder_act_upsampling_only'
]


def construct_model(conf, model_name, **kwargs):
  params = conf.to_param_dict(REQUIRED_PARAMS, OPTIONAL_PARAMS)
  model = UNET(**params)
  initialize_weights(model, conf.get_attr('weight_init', default={}))
  return model


def _pad_to_target(x, target, mode='reflect'):
  _, _, h, w = x.size()
  _, _, h2, w2 = target.size()
  pad_bottom = h2 - h
  pad_right = w2 - w
  if pad_bottom != 0 or pad_right != 0:
    x = F.pad(x, (0, pad_right, 0, pad_bottom), mode=mode)
  return x


class ConvEncodeUnit(nn.Module):
  def __init__(self, in_channels, num_layers, num_filters, kernel_size,
               relu_leakiness, use_bn, downsample,
               use_act=True, padding='zero'):
    super(ConvEncodeUnit, self).__init__()
    self.downsample = downsample
    use_bias = not use_bn

    modules = []
    for i in range(num_layers):
      modules += [get_same_padding_layer(kernel_size, stride=1, mode=padding),
                  nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size,
                            stride=1, bias=use_bias)]
      in_channels = num_filters
      if use_bn:
        modules += [nn.BatchNorm2d(in_channels)]
      if use_act:
        modules += [nn.LeakyReLU(relu_leakiness, inplace=True)]

    self.encode = nn.Sequential(*modules)
    if downsample:
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, inp):
    x = self.encode(inp)
    if self.downsample:
      x_before = x
      x = self.pool(x_before)
      return x, x_before
    else:
      return x


class ConvDecodeUnit(nn.Module):
  def __init__(self, in_channels, encoder_channels, num_filters,
               relu_leakiness, use_bn, use_act=True,
               kernel_size=3, transposed_kernel_size=2, num_layers=0,
               mode='transposed', padding='zero',
               act_upsampling_only=False):
    super(ConvDecodeUnit, self).__init__()
    assert mode in ('transposed', 'nn', 'bilinear', 'pixelshuffle',
                    'nn-resize-conv', 'nn-biresize-conv')
    use_bias = not use_bn or encoder_channels == 0

    if mode == 'transposed':
      upsample = [nn.ConvTranspose2d(in_channels, num_filters,
                                     kernel_size=transposed_kernel_size,
                                     stride=2, bias=use_bias)]
      in_channels = num_filters
    elif mode == 'nn':
      upsample = [nn.Upsample(scale_factor=2, mode='nearest')]
    elif mode == 'bilinear':
      upsample = [nn.Upsample(scale_factor=2, mode='bilinear')]
    elif mode == 'pixelshuffle':
      upsample = [get_same_padding_layer(kernel_size, stride=1, mode=padding),
                  nn.Conv2d(in_channels, 4 * num_filters,
                            kernel_size=kernel_size, stride=1, bias=use_bias),
                  nn.PixelShuffle(upscale_factor=2)]
      in_channels = num_filters
    elif mode == 'nn-resize-conv' or mode == 'nn-biresize-conv':
      resize_mode = 'nearest' if mode == 'nn-resize-conv' else 'bilinear'
      upsample = [nn.Upsample(scale_factor=2, mode=resize_mode),
                  get_same_padding_layer(kernel_size, stride=1, mode=padding),
                  nn.Conv2d(in_channels, num_filters,
                            kernel_size=kernel_size, stride=1, bias=use_bias)]
      in_channels = num_filters

    decode = []

    if act_upsampling_only:
      # Add batch norm and activation only on the upsampling path:
      # This way saves some computation+memory by not activating on the
      # encoded features again
      if use_bn:
        upsample += [nn.BatchNorm2d(in_channels)]
      if use_act:
        upsample += [nn.LeakyReLU(relu_leakiness, inplace=True)]
    else:
      # Add batch norm and activation for the concatenated features from both
      # encoder path and upsampling path. Is kept here for legacy reasons.
      if use_bn:
        decode += [nn.BatchNorm2d(in_channels + encoder_channels)]
      if use_act:
        decode += [nn.LeakyReLU(relu_leakiness, inplace=True)]

    if num_layers > 0:
      decode += [ConvEncodeUnit(in_channels + encoder_channels, num_layers,
                                num_filters, kernel_size, relu_leakiness,
                                use_bn, downsample=False, use_act=use_act,
                                padding=padding)]

    self.upsample = nn.Sequential(*upsample)
    self.decode = nn.Sequential(*decode)

  def forward(self, decode_path, encode_path=None):
    x = self.upsample(decode_path)
    if encode_path is not None:
      x = _pad_to_target(x, encode_path)
      x = torch.cat((encode_path, x), dim=1)
    return self.decode(x)


class UNET(nn.Module):
  DEFAULT_RELU_LEAKINESS = 0.1

  def __init__(self, num_inputs, num_outputs, num_layers_per_scale,
               encode_filters, decode_filters, output_activation,
               kernel_size=3, transposed_kernel_size=2,
               relu_leakiness=DEFAULT_RELU_LEAKINESS,
               use_bn=True, upsampling_mode='transposed', padding='zero',
               encoder_features=None, use_refinement=False,
               decoder_act_upsampling_only=False):
    """Build a UNET

    Parameters
    ----------
    num_inputs : int
      Number of input channels
    num_outputs : int
      Number of output channels
    num_layers_per_scale : int
      Number of convolutional layers per scale
    encode_filters : list
      Number of filters per scale the network uses in the encode path
    decode_filters : list
      Number of filters per scale the network uses in the decode path
    output_activation : string
      Either `softmax`, `tanh` or `none`. Activation function to use
      on the logits
    kernel_size : int
      Convolutional filter size
    transposed_kernel_size : int
      Convolutional filter size of transposed convolution, if using
      upsampling_mode `transposed`
    relu_leakiness : float or tuple of floats
      If tuple, leakiness of the relus for encode and decode path.
      If float, use the same leakiness for encode and decode path
    use_bn : bool
      If true, use batch norm layers after each convolution
    upsampling_mode : string
      Upsampling method to use. Either `transposed` (default), `bilinear`
      or `pixelshuffle`, `nn-resize-conv`, `nn-biresize-conv`
    padding : string
      Type of padding to use. Either `zero`, `reflection`, or `replication`
    encoder_features : list of int
      If not None, indices of encoder features to return under key `features`
    use_refinement : bool
      If true, learn an additive transformation with respect to the input
      image
    decoder_act_upsampling_only : bool
      If true, use batch norm and activation function only on the upsampled
      features, not on both encoded and upsampled features. Better option,
      but disabled by default for backwards compatibility
    """
    super(UNET, self).__init__()
    assert output_activation in ('softmax', 'tanh', 'none')

    self.encoder_features = encoder_features
    self.use_refinement = use_refinement

    if isinstance(relu_leakiness, float):
      relu_leakiness = (relu_leakiness, relu_leakiness)

    in_channels = num_inputs

    num_encode_units = len(encode_filters)
    encode_channels = []
    encode_units = []
    for scale, num_filters in enumerate(encode_filters):
      downsample = scale != len(encode_filters) - 1
      unit = ConvEncodeUnit(in_channels, num_layers_per_scale, num_filters,
                            kernel_size, relu_leakiness[0], use_bn,
                            downsample=downsample, padding=padding)
      encode_units.append(unit)
      encode_channels.append(num_filters)
      in_channels = num_filters

    concat_decode_units = []
    for scale, num_filters in enumerate(decode_filters[:num_encode_units - 1]):
      num_encode_channels = encode_channels[-(scale + 2)]
      unit = ConvDecodeUnit(in_channels, num_encode_channels,
                            num_filters, relu_leakiness[1], use_bn,
                            kernel_size=kernel_size,
                            transposed_kernel_size=transposed_kernel_size,
                            num_layers=num_layers_per_scale,
                            mode=upsampling_mode, padding=padding,
                            act_upsampling_only=decoder_act_upsampling_only)
      concat_decode_units.append(unit)
      in_channels = num_filters

    decode_units = []
    for scale, num_filters in enumerate(decode_filters[num_encode_units - 1:]):
      unit = ConvDecodeUnit(in_channels, 0, num_filters, relu_leakiness[1],
                            use_bn,
                            kernel_size=kernel_size,
                            transposed_kernel_size=transposed_kernel_size,
                            num_layers=num_layers_per_scale,
                            mode=upsampling_mode,
                            padding=padding,
                            act_upsampling_only=decoder_act_upsampling_only)
      decode_units.append(unit)
      in_channels = num_filters

    head = []
    head += [nn.Conv2d(in_channels, num_outputs, kernel_size=1,
                       stride=1, padding=0, bias=True)]
    if output_activation == 'softmax':
      head += [nn.Softmax()]
    elif output_activation == 'tanh':
      head += [nn.Tanh()]

    self.encode_units = nn.ModuleList(encode_units)
    self.concat_decode_units = nn.ModuleList(concat_decode_units)
    self.decode_units = nn.ModuleList(decode_units)
    self.head = nn.Sequential(*head)

  @staticmethod
  def weight_init_params(user_weight_init=None):
    return {
        'conv_weight': ('he_normal', UNET.DEFAULT_RELU_LEAKINESS),
        'conv_transposed_weight': ('he_normal', UNET.DEFAULT_RELU_LEAKINESS),
        'batchnorm_weight': ('uniform', 0.98, 1.02)
    }

  def forward(self, inp):
    x = inp
    encoder_features = []
    for scale, unit in enumerate(self.encode_units):
      if unit.downsample:
        x, features = unit(x)
        encoder_features.append(features)
      else:
        x = unit(x)
        encoder_last_feature = x

    for scale, unit in enumerate(self.concat_decode_units):
      x = unit(x, encoder_features[-(scale + 1)])

    for scale, unit in enumerate(self.decode_units):
      x = unit(x)

    pred = self.head(x)

    if self.use_refinement:
      pred = inp + pred

    if self.encoder_features is not None:
      encoded_features = encoder_features + [encoder_last_feature]
      return {
          'pred': pred,
          'features': [encoded_features[idx] for idx in self.encoder_features]
      }
    else:
      return pred
