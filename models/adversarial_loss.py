import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_adversarial_loss(conf, loss_name, cuda, loss_type):
  assert loss_type == 'disc' or loss_type == 'gen'

  disc_label_smoothing = conf.get_attr('discriminator_label_smoothing',
                                       default=0.0)

  if loss_name.upper() == 'GAN':
    return GANLoss(loss_type, cuda, disc_label_smoothing)
  elif loss_name.upper() == 'LSGAN':
    return LeastSquaresLoss(loss_type, cuda, disc_label_smoothing)
  elif loss_name.upper() == 'WGAN':
    return WGANLoss(loss_type)
  elif loss_name == 'FeatureMatching' or loss_name == 'feature-matching':
    distance_fn = conf.get_attr('feature_matching_loss_distance_function',
                                default='L1')
    return FeatureMatchingLoss(loss_type, distance_fn)
  else:
    raise ValueError('Unknown loss {}'.format(loss_name))


class _AdversarialLoss(nn.Module):
  def __init__(self, loss_type, cuda, loss_fn, disc_label_smoothing=0.0):
    super(_AdversarialLoss, self).__init__()
    assert loss_type in ('disc', 'gen'), \
        'Unknown adversarial loss type {}'.format(loss_type)
    assert 0.0 <= disc_label_smoothing < 1.0
    self.loss_fn = loss_fn

    # Setting this to 1 means that the generator maximizes the probability
    # that the discriminator assigns label 1 to generated images
    self.gen_label = 1.0
    self.gen_real_var = None
    self.disc_real_label = 1.0 - disc_label_smoothing
    self.disc_real_var = None
    self.disc_fake_label = 0.0
    self.disc_fake_var = None

    if cuda != '':
      self.tensor_fn = lambda *args: torch.FloatTensor(*args).cuda()
    else:
      self.tensor_fn = lambda *args: torch.FloatTensor(*args)

    # Set if loss is for generator or discriminator
    if loss_type == 'gen':
      self.forward = self.loss_gen
    else:
      self.forward = self.loss_disc

  def _get_label_var(self, prev_label_var, shape, label):
    """Get the loss target label

    Avoids creating the label variable more than once if the shape does not
    change.
    """
    if prev_label_var is None or prev_label_var.shape != shape:
      tensor = self.tensor_fn(shape).fill_(label)
      return Variable(tensor, requires_grad=False)
    else:
      return prev_label_var

  def _loss_disc(self, pred_fake, pred_real):
    self.disc_fake_var = self._get_label_var(self.disc_fake_var,
                                             pred_fake.shape,
                                             self.disc_fake_label)
    loss_fake = self.loss_fn(pred_fake, self.disc_fake_var)

    self.disc_real_var = self._get_label_var(self.disc_real_var,
                                             pred_real.shape,
                                             self.disc_real_label)
    loss_real = self.loss_fn(pred_real, self.disc_real_var)

    return loss_fake + loss_real

  def _loss_gen(self, pred_fake):
    self.gen_real_var = self._get_label_var(self.gen_real_var,
                                            pred_fake.shape,
                                            self.gen_label)
    loss_fake = self.loss_fn(pred_fake, self.gen_real_var)
    return loss_fake


class GANLoss(_AdversarialLoss):
  def __init__(self, loss_type, cuda, disc_label_smoothing):
    super(GANLoss, self).__init__(loss_type, cuda, F.binary_cross_entropy,
                                  disc_label_smoothing)

  def loss_disc(self, out_disc_fake, out_disc_real):
    return self._loss_disc(out_disc_fake['prob'],
                           out_disc_real['prob'])

  def loss_gen(self, out_disc_fake, out_disc_real):
    return self._loss_gen(out_disc_fake['prob'])


class LeastSquaresLoss(_AdversarialLoss):
  def __init__(self, loss_type, cuda, disc_label_smoothing):
    super(LeastSquaresLoss, self).__init__(loss_type, cuda, F.mse_loss,
                                           disc_label_smoothing)

  def loss_disc(self, out_disc_fake, out_disc_real):
    return self._loss_disc(out_disc_fake['logits'],
                           out_disc_real['logits'])

  def loss_gen(self, out_disc_fake, out_disc_real):
    return self._loss_gen(out_disc_fake['logits'])


class WGANLoss(nn.Module):
  def __init__(self, loss_type):
    super(WGANLoss, self).__init__()

    # Set if loss is for generator or discriminator
    if loss_type == 'gen':
      self.forward = self.loss_gen
    else:
      self.forward = self.loss_disc

  def loss_disc(self, out_disc_fake, out_disc_real):
    return out_disc_fake['logits'].mean() - out_disc_real['logits'].mean()

  def loss_gen(self, out_disc_fake, out_disc_real):
    return -out_disc_fake['logits'].mean()


class FeatureMatchingLoss(nn.Module):
  def __init__(self, loss_type, distance_fn):
    super(FeatureMatchingLoss, self).__init__()
    distance_fns = {
        'MSE': F.mse_loss,
        'L1': F.l1_loss
    }
    assert distance_fn in distance_fns, \
        'Unknown distance function {}'.format(distance_fn)

    self.distance_fn = distance_fns[distance_fn]

    # Set if loss is for generator or discriminator
    if loss_type == 'gen':
      self.forward = self.loss_gen
    else:
      self.forward = self.loss_disc

  def loss_disc(self, out_disc_fake, out_disc_real):
    return -1. * self.loss_gen(out_disc_fake, out_disc_real)

  def loss_gen(self, out_disc_fake, out_disc_real):
    features_fake = out_disc_fake['features']
    features_real = out_disc_real['features']

    loss = 0
    for f_fake, f_real in zip(features_fake, features_real):
      loss += self.distance_fn(f_fake, f_real.detach())

    return loss / len(features_fake)
