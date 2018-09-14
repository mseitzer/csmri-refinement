from collections import OrderedDict
import logging

import torch

import utils
from data.transform_wrappers import (get_output_transform,
                                     get_input_batch_transform)
from metrics import get_metric_fn, get_loss_metric, accumulate_metric
from models import construct_model
from models.criteria import get_criterion
from training.adversarial_training import get_discriminator_input_fn
from training.lr_schedulers import (get_lr_scheduler,
                                    is_pre_epoch_scheduler,
                                    is_post_epoch_scheduler)
from training.optimizers import get_optimizer
from training.base_runner import BaseRunner
from utils.checkpoints import initialize_pretrained_model
from utils.config import Configuration


def build_runner(conf, cuda, mode):
  gen_model_conf = Configuration.from_dict(conf.generator_model, conf)
  gen_model = construct_model(gen_model_conf, gen_model_conf.name, cuda)

  val_metric_fns = {name: get_metric_fn(conf, name, cuda, 'test')
                    for name in conf.get_attr('validation_metrics',
                                              default=[])}
  output_transform = get_output_transform(conf, conf.application, 'inference')
  test_input_batch_transform = get_input_batch_transform(conf,
                                                         conf.application,
                                                         'test')

  if mode == 'train':
    disc_model_conf = Configuration.from_dict(conf.discriminator_model, conf)
    disc_model = construct_model(disc_model_conf, disc_model_conf.name, cuda)

    gen_adv_criteria = {loss_name: get_criterion(conf, loss_name, cuda,
                                                 loss_type='gen')
                        for loss_name in conf.generator_adversarial_losses}
    gen_criteria = {loss_name: get_criterion(conf, loss_name, cuda)
                    for loss_name in conf.generator_losses}
    disc_adv_criteria = {loss_name: get_criterion(conf, loss_name, cuda,
                                                  loss_type='disc')
                         for loss_name in conf.discriminator_losses}

    if cuda != '':
      # Potentially split models over GPUs
      gen_model, disc_model = utils.cudaify([gen_model, disc_model], cuda)
      utils.cudaify(list(gen_adv_criteria.values()) +
                    list(gen_criteria.values()) +
                    list(disc_adv_criteria.values()))

    # Important: construct optimizers after moving model to GPU!
    gen_opt_conf = Configuration.from_dict(conf.generator_optimizer, conf)
    gen_optimizer = get_optimizer(gen_opt_conf, gen_opt_conf.name,
                                  gen_model.parameters())
    gen_lr_scheduler = None
    if gen_opt_conf.has_attr('lr_scheduler'):
      gen_lr_scheduler = get_lr_scheduler(gen_opt_conf,
                                          gen_opt_conf.lr_scheduler,
                                          gen_optimizer)

    disc_opt_conf = Configuration.from_dict(conf.discriminator_optimizer, conf)
    disc_optimizer = get_optimizer(disc_opt_conf, disc_opt_conf.name,
                                   disc_model.parameters())
    disc_lr_scheduler = None
    if disc_opt_conf.has_attr('lr_scheduler'):
      disc_lr_scheduler = get_lr_scheduler(disc_opt_conf,
                                           disc_opt_conf.lr_scheduler,
                                           disc_optimizer)

    train_input_batch_transform = get_input_batch_transform(conf,
                                                            conf.application,
                                                            'train')
    train_disc_metrics = conf.get_attr('train_discriminator_metrics',
                                       default=[])
    train_disc_metric_fns = {name: get_metric_fn(conf, name, cuda, 'train')
                             for name in train_disc_metrics}
    val_disc_metric_key = 'validation_discriminator_metrics'
    val_disc_metric_fns = {name: get_metric_fn(conf, name, cuda, 'test')
                           for name in conf.get_attr(val_disc_metric_key,
                                                     default=[])}

    train_gen_metrics = conf.get_attr('train_generator_metrics', default=[])
    train_gen_metric_fns = {name: get_metric_fn(conf, name, cuda, 'train')
                            for name in train_gen_metrics}

    disc_input_fn = get_discriminator_input_fn(conf, disc_model_conf)
    val_disc_input_fn = get_discriminator_input_fn(conf, disc_model_conf,
                                                   no_pool=True)

    pretr_generator_epochs = conf.get_attr('pretrain_generator_epochs')
    pretr_discriminator_epochs = conf.get_attr('pretrain_discriminator_epochs')

    runner = AdversarialRunner(gen_model, disc_model,
                               gen_optimizer, disc_optimizer,
                               gen_lr_scheduler, disc_lr_scheduler,
                               gen_adv_criteria, gen_criteria,
                               disc_adv_criteria,
                               conf.get_attr('generator_loss_weights', {}),
                               conf.get_attr('discriminator_loss_weights', {}),
                               cuda,
                               train_gen_metric_fns,
                               train_disc_metric_fns,
                               val_metric_fns,
                               val_disc_metric_fns,
                               output_transform,
                               train_input_batch_transform,
                               test_input_batch_transform,
                               gen_opt_conf.get_attr('updates_per_step', 1),
                               disc_opt_conf.get_attr('updates_per_step', 1),
                               disc_input_fn,
                               val_disc_input_fn,
                               pretr_generator_epochs,
                               pretr_discriminator_epochs)
    if gen_model_conf.has_attr('pretrained_weights'):
      initialize_pretrained_model(gen_model_conf, runner.gen, cuda, conf.file)

    if disc_model_conf.has_attr('pretrained_weights'):
      initialize_pretrained_model(disc_model_conf, runner.disc, cuda,
                                  conf.file)
  else:
    if cuda != '':
      utils.cudaify(gen_model)
    runner = AdversarialRunner(gen_model,
                               cuda=cuda,
                               val_metric_fns=val_metric_fns,
                               output_transform=output_transform,
                               test_input_batch_transform=test_input_batch_transform)

  return runner


class AdversarialRunner(BaseRunner):
  """A runner for an adversarial model with generator and discriminator"""
  def __init__(self, gen_model, disc_model=None,
               gen_optimizer=None, disc_optimizer=None,
               gen_lr_scheduler=None, disc_lr_scheduler=None,
               gen_adv_criteria={}, gen_criteria={}, disc_adv_criteria={},
               gen_loss_weights={}, disc_loss_weights={}, cuda='',
               train_gen_metric_fns={}, train_disc_metric_fns={},
               val_metric_fns={}, val_disc_metric_fns={},
               output_transform=None,
               train_input_batch_transform=None,
               test_input_batch_transform=None,
               gen_updates_per_step=1, disc_updates_per_step=1,
               disc_input_fn=None,
               val_disc_input_fn=None,
               pretrain_generator_epochs=None,
               pretrain_discriminator_epochs=None):
    super(AdversarialRunner, self).__init__(cuda)
    self.gen = gen_model
    self.disc = disc_model
    self.gen_optimizer = gen_optimizer
    self.disc_optimizer = disc_optimizer
    self.gen_lr_scheduler = gen_lr_scheduler
    self.disc_lr_scheduler = disc_lr_scheduler

    self.train_gen_metric_fns = train_gen_metric_fns
    self.train_disc_metric_fns = train_disc_metric_fns
    self.val_metric_fns = val_metric_fns
    self.val_disc_metric_fns = val_disc_metric_fns
    self.output_transform = output_transform
    input_fn = self._get_model_input_fn(self.gen, train_input_batch_transform)
    self.train_model_input_fn = input_fn
    input_fn = self._get_model_input_fn(self.gen, test_input_batch_transform)
    self.test_model_input_fn = input_fn

    self.gen_updates_per_step = gen_updates_per_step
    self.disc_updates_per_step = disc_updates_per_step

    self.disc_input_fn = disc_input_fn
    self.val_disc_input_fn = val_disc_input_fn

    # Set train function according to update steps
    if gen_updates_per_step == 1 and disc_updates_per_step == 1:
      self._train_step = self._train_single_step
    else:
      self._train_step = self._train_multiple_steps

    self.gen_adv_criteria = OrderedDict(gen_adv_criteria)
    self.gen_criteria = OrderedDict(gen_criteria)
    self.disc_adv_criteria = OrderedDict(disc_adv_criteria)

    self.gen_loss_weights = self._get_loss_weights(gen_loss_weights,
                                                   gen_adv_criteria,
                                                   gen_criteria)
    self.disc_loss_weights = self._get_loss_weights(disc_loss_weights,
                                                    disc_adv_criteria)

    self.discriminator_enabled = True
    self.generator_enabled = True

    def _get_pretraining_schedule(epochs):
      if epochs is None:
        return (-1, -1)
      elif isinstance(epochs, int):
        return (1, epochs + 1)
      else:
        # epochs is an interval
        assert epochs[0] < epochs[1],  \
            'Starting epoch must be smaller than ending epoch'
        return epochs

    schedule = _get_pretraining_schedule(pretrain_generator_epochs)
    self.generator_pretraining_schedule = schedule
    schedule = _get_pretraining_schedule(pretrain_discriminator_epochs)
    self.discriminator_pretraining_schedule = schedule

  def get_named_outputs(self, data):
    batch, out_gen = data[0], data[1]
    target = batch['target']

    if isinstance(out_gen, dict):
      prediction = out_gen['pred']
    else:
      prediction = out_gen

    if self.output_transform is not None:
      prediction, target = self.output_transform(prediction, target)

    return {
        'input': batch['inp'],
        'prediction': prediction,
        'target': target,
        'disc_fake': data[2]
    }

  def get_named_models(self):
    return {
        'generator': self.gen,
        'discriminator': self.disc,
    }

  def state_dict(self):
    return {
        'generator': self.gen.state_dict(),
        'discriminator': self.disc.state_dict(),
        'gen_optimizer': self.gen_optimizer.state_dict(),
        'disc_optimizer': self.disc_optimizer.state_dict()
    }

  def load_state_dict(self, state_dict):
    self.gen.load_state_dict(state_dict['generator'])

    if self.disc is not None:
      assert 'discriminator' in state_dict, 'Incompatible checkpoint'
      self.disc.load_state_dict(state_dict['discriminator'])

    if self.gen_optimizer is not None:
      assert 'gen_optimizer' in state_dict, 'Incompatible checkpoint'
      self.gen_optimizer.load_state_dict(state_dict['gen_optimizer'])

    if self.disc_optimizer is not None:
      assert 'disc_optimizer' in state_dict, 'Incompatible checkpoint'
      self.disc_optimizer.load_state_dict(state_dict['disc_optimizer'])

  def __str__(self):
    s = 'Generator:\n'
    s += str(self.gen)
    if self.disc is not None:
      s += '\nDiscriminator:\n'
      s += str(self.disc)
    return s

  def epoch_beginning(self, epoch):
    if is_pre_epoch_scheduler(self.gen_lr_scheduler):
      self.gen_lr_scheduler.step()
    if is_pre_epoch_scheduler(self.disc_lr_scheduler):
      self.disc_lr_scheduler.step()

    start, end = self.generator_pretraining_schedule
    if start <= epoch < end:
      logging.debug('Pretraining generator, discriminator disabled')
      self.discriminator_enabled = False
      self.generator_enabled = True
    else:
      self.discriminator_enabled = True

    if start == epoch:
      logging.info('Start pretraining generator in epoch {}'.format(epoch))
    elif end == epoch:
      logging.info('Stop pretraining generator before epoch {}'.format(epoch))

    start, end = self.discriminator_pretraining_schedule
    if start <= epoch < end:
      logging.debug('Pretraining discriminator, generator disabled')
      self.discriminator_enabled = True
      self.generator_enabled = False
    else:
      self.generator_enabled = True

    if start == epoch:
      logging.info(('Start pretraining discriminator '
                    'in epoch {}').format(epoch))
    elif end == epoch:
      logging.info(('Stop pretraining discriminator '
                    'before epoch {}').format(epoch))

  def epoch_finished(self, epoch):
    if is_post_epoch_scheduler(self.gen_lr_scheduler):
      self.gen_lr_scheduler.step()
    if is_post_epoch_scheduler(self.disc_lr_scheduler):
      self.disc_lr_scheduler.step()

  def predict(self, batch):
    # Note: for now, we are not differentiating between train and validation
    # predict, so we are using train_model_input_fn without transform
    # for both cases.
    return self.gen(*self.train_model_input_fn(batch,
                                               use_batch_transform=False))

  @staticmethod
  def _update_step(optimizer, losses, weights):
    total_loss = torch.sum(torch.cat(losses) * weights)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss

  def _train_single_step(self, loader):
    batch = self._request_data(loader)
    if batch is None:
      return 0, None, None

    loss_metrics = {}
    gen_inp = self.train_model_input_fn(batch)
    out_gen = self.gen(*gen_inp)

    if self.discriminator_enabled:
      # Propagate fake image through discriminator
      out_disc_fake = self.disc(self.disc_input_fn(out_gen, gen_inp[0], out_gen,
                                                   is_real_input=False,
                                                   detach=True))

      # Propagate real images through discriminator
      target = batch['target']
      out_disc_real = self.disc(self.disc_input_fn(target, gen_inp[0], out_gen,
                                                   is_real_input=True,
                                                   detach=True))
      disc_losses = []
      # Compute discriminator losses
      for name, criterion in self.disc_adv_criteria.items():
        loss = criterion(out_disc_fake, out_disc_real)
        disc_losses.append(loss)
        loss_metrics['disc_loss_' + name] = get_loss_metric(loss.data[0])

    if self.generator_enabled:
      gen_losses = []
      if self.discriminator_enabled:
        # Propagate again with non-detached input to allow gradients on the
        # generator
        out_disc_fake = self.disc(self.disc_input_fn(out_gen, gen_inp[0],
                                                     out_gen,
                                                     is_real_input=False,
                                                     detach=False))
        # Compute adversarial generator losses from discriminator output
        # Order matters: first compute adversarial losses for generator, then
        # the other generator losses. Otherwise the loss weights will not match
        for name, criterion in self.gen_adv_criteria.items():
          loss = criterion(out_disc_fake, out_disc_real)
          gen_losses.append(loss)
          loss_metrics['gen_loss_' + name] = get_loss_metric(loss.data[0])

      # Compute generator losses on prediction and target image
      for name, criterion in self.gen_criteria.items():
        loss = criterion(out_gen, batch)
        gen_losses.append(loss)
        loss_metrics['gen_loss_' + name] = get_loss_metric(loss.data[0])

    # Perform updates
    if self.discriminator_enabled:
      total_disc_loss = self._update_step(self.disc_optimizer,
                                          disc_losses,
                                          self.disc_loss_weights)
      loss_metrics['disc_loss'] = get_loss_metric(total_disc_loss.data[0])

    if self.generator_enabled:
      total_gen_loss = self._update_step(self.gen_optimizer,
                                         gen_losses,
                                         self.gen_loss_weights)
      loss_metrics['gen_loss'] = get_loss_metric(total_gen_loss.data[0])

    if not self.discriminator_enabled:
        out_disc_fake = None
        out_disc_real = None

    return 1, loss_metrics, (batch, out_gen, out_disc_fake, out_disc_real)

  def _train_multiple_steps(self, loader):
    """Train generator and discriminator for multiple steps at once"""
    last_batch = None
    max_updates = max(self.disc_updates_per_step,
                      self.gen_updates_per_step)

    # Deque input data upfront (this could lead to memory problems)
    batches = []
    for _ in range(max_updates):
      batch = self._request_data(loader)
      if batch is None:
        break
      batches.append(batch)

    gen_uses_feature_matching = 'FeatureMatching' in self.gen_adv_criteria
    loss_metrics = {}

    # Train discriminator
    for idx, batch in enumerate(batches[:self.disc_updates_per_step]):
      if not self.discriminator_enabled:
        continue

      last_batch = batch

      # Propagate fake image through discriminator
      gen_inp = self.train_model_input_fn(batch)
      out_gen = self.gen(*gen_inp)
      out_disc_fake = self.disc(self.disc_input_fn(out_gen, gen_inp[0],
                                                   out_gen,
                                                   is_real_input=False,
                                                   detach=True))

      # Propagate real images through discriminator
      target = batch['target']
      out_disc_real = self.disc(self.disc_input_fn(target, gen_inp[0], out_gen,
                                                   is_real_input=True,
                                                   detach=True))

      disc_losses = []
      # Compute discriminator losses
      for name, criterion in self.disc_adv_criteria.items():
        loss = criterion(out_disc_fake, out_disc_real)
        disc_losses.append(loss)
        accumulate_metric(loss_metrics, 'disc_loss_' + name,
                          get_loss_metric(loss.data[0]))

      # Perform discriminator update
      total_disc_loss = self._update_step(self.disc_optimizer,
                                          disc_losses,
                                          self.disc_loss_weights)
      accumulate_metric(loss_metrics, 'disc_loss',
                        get_loss_metric(total_disc_loss.data[0]))

      if idx < len(batches) - 1 and idx < self.disc_updates_per_step - 1:
        del out_gen
        del out_disc_real
        del out_disc_fake
      elif self.generator_enabled:
        del out_gen
        del out_disc_fake

    # Train generator
    for idx, batch in enumerate(batches[:self.gen_updates_per_step]):
      if not self.generator_enabled:
        continue

      last_batch = batch
      gen_losses = []

      gen_inp = self.train_model_input_fn(batch)
      out_gen = self.gen(*gen_inp)

      if self.discriminator_enabled:
        # Propagate again with non-detached input to allow gradients on the
        # generator
        out_disc_fake = self.disc(self.disc_input_fn(out_gen, gen_inp[0],
                                                     out_gen,
                                                     is_real_input=False,
                                                     detach=False))
        if gen_uses_feature_matching:
          # Only need to compute the discriminator output for real targets if we
          # use feature matching loss
          target = batch['target']
          out_disc_real = self.disc(self.disc_input_fn(target, gen_inp[0],
                                                       out_gen,
                                                       is_real_input=True,
                                                       detach=True))
        else:
          out_disc_real = None

        # Compute adversarial generator losses from discriminator output
        # Order matters: first compute adversarial losses for generator, then
        # the other generator losses. Otherwise the loss weights will not match
        for name, criterion in self.gen_adv_criteria.items():
          loss = criterion(out_disc_fake, out_disc_real)
          gen_losses.append(loss)
          accumulate_metric(loss_metrics, 'gen_loss_' + name,
                            get_loss_metric(loss.data[0]))

      # Compute generator losses on prediction and target image
      for name, criterion in self.gen_criteria.items():
        loss = criterion(out_gen, batch)
        gen_losses.append(loss)
        accumulate_metric(loss_metrics, 'gen_loss_' + name,
                          get_loss_metric(loss.data[0]))

      # Perform generator update
      total_gen_loss = self._update_step(self.gen_optimizer,
                                         gen_losses,
                                         self.gen_loss_weights)
      accumulate_metric(loss_metrics, 'gen_loss',
                        get_loss_metric(total_gen_loss.data[0]))

      if idx < len(batch) - 1 and idx < self.gen_updates_per_step - 1:
        del out_gen
        if self.discriminator_enabled:
          del out_disc_fake
          if out_disc_real is not None:
            del out_disc_real

    # For simplicity, we just return the last batch of data
    # This is a bit of a smell, as our metrics will only be on this last batch
    # of data, whereas the loss metrics are averaged over all updates
    if len(batches) > 0:
      avg_loss_metrics = {name: metric.average()
                          for name, metric in loss_metrics.items()}
      if not self.discriminator_enabled:
        out_disc_fake = None
        out_disc_real = None
      data = (last_batch, out_gen, out_disc_fake, out_disc_real)
    else:
      avg_loss_metrics = None
      data = None

    return len(batches), avg_loss_metrics, data

  def _val_step(self, loader, compute_metrics=True):
    batch = self._request_data(loader, volatile=True)
    if batch is None:
      return None, None

    gen_inp = self.test_model_input_fn(batch)
    out_gen = self.gen(*gen_inp)

    if self.disc is not None:
      out_disc_fake = self.disc(self.val_disc_input_fn(out_gen, gen_inp[0],
                                                       out_gen,
                                                       is_real_input=False,
                                                       detach=True))
      target = batch['target']
      out_disc_real = self.disc(self.val_disc_input_fn(target, gen_inp[0],
                                                       out_gen,
                                                       is_real_input=True,
                                                       detach=True))
    else:
      out_disc_fake = None
      out_disc_real = None

    loss_metrics = {}
    if compute_metrics:
      # Only compute the standard losses here, adversarial losses don't make
      # to much sense
      for name, criterion in self.gen_criteria.items():
        loss = criterion(out_gen, batch)
        loss_metrics['gen_loss_' + name] = get_loss_metric(loss.data[0])

    return loss_metrics, (batch, out_gen, out_disc_fake, out_disc_real)

  def _compute_gen_metrics(self, metrics, metric_fns, predictions, targets):
    for metric_name, metric_fn in metric_fns.items():
      metric = metric_fn(predictions, targets)
      metrics['gen_' + metric_name] = metric

    return metrics

  def _compute_disc_metrics(self, metrics, metric_fns,
                            out_disc_fake, out_disc_real):
    prob_fake = out_disc_fake['prob']
    prob_real = out_disc_real['prob']

    for metric_name, metric_fn in metric_fns.items():
      metric = metric_fn(prob_fake, prob_real, transform=False)
      metrics['disc_' + metric_name] = metric

    return metrics

  def _compute_train_metrics(self, data):
    metrics = {}
    _ = self._compute_gen_metrics(metrics,
                                  self.train_gen_metric_fns,
                                  data[1], data[0])
    if data[2] is not None:
      _ = self._compute_disc_metrics(metrics,
                                     self.train_disc_metric_fns,
                                     data[2], data[3])
    return metrics

  def _compute_test_metrics(self, data):
    metrics = {}
    _ = self._compute_gen_metrics(metrics,
                                  self.val_metric_fns,
                                  data[1], data[0])
    if data[2] is not None:
      _ = self._compute_disc_metrics(metrics,
                                     self.val_disc_metric_fns,
                                     data[2], data[3])
    return metrics

  def _set_train(self):
    self.gen.train()
    self.disc.train()

  def _set_test(self):
    self.gen.eval()
    if self.disc is not None:
      self.disc.eval()
