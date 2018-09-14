#!/usr/bin/env python
import argparse
import logging
import math
import os
import sys
import time
from itertools import chain

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import utils
from data import load_dataset, maybe_get_subset_sampler
from utils.config import Configuration
from utils.diagnostics import print_model_parameters
from training import build_runner
from utils.checkpoints import (save_checkpoint,
                               restore_checkpoint,
                               prune_checkpoints)
from utils.checkpoint_paths import (get_run_dir,
                                    get_config_path,
                                    get_periodic_checkpoint_path,
                                    get_best_checkpoint_path)
from utils.logging import setup_logging

DEFAULT_EPOCHS_PER_CHECKPOINT = 5
DEFAULT_EPOCHS_PER_VALIDATION = 5
DEFAULT_STEPS_PER_TRAIN_SUMMARY = 1
DEFAULT_NUM_WORKERS = 2
DEFAULT_NUM_PERIODIC_CHECKPOINTS = 1
DEFAULT_NUM_BEST_CHECKPOINTS = 3
DEFAULT_USE_TENSORBOARD = False
DEFAULT_NUM_IMAGE_SUMMARIES = 0
DEFAULT_INITIAL_VALIDATION = False
DEFAULT_BEST_VALUE_WARMUP_EPOCHS = 0

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('-c', '--cuda', default='0', type=str, help='GPU to use')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print more info')
parser.add_argument('-p', '--print-model', action='store_true',
                    help='Print model informations')
parser.add_argument('--print-parameters', action='store_true',
                    help='Print parameter information')
parser.add_argument('--dry', action='store_true',
                    help=('Do not create output directories. '
                          'Useful for debugging'))
parser.add_argument('--conf', nargs='+',
                    help='Optional config values to set')
parser.add_argument('--data-dir', default='resources/data',
                    help='Path to data directory')
parser.add_argument('--log-dir', default='resources/models',
                    help='Path to log directory')
parser.add_argument('--run-dir',
                    help='Path to specific output directory')
parser.add_argument('--resume',
                    help='Path to a checkpoint to resume training from')
parser.add_argument('config', help='Config file to use')


def save_periodic_checkpoint(conf, runner, epoch, best_val_metrics):
  log_file_path = get_periodic_checkpoint_path(conf.run_dir, epoch)
  if not os.path.isdir(os.path.dirname(log_file_path)):
    logging.warning(('Skip saving periodic checkpoint: {} does not '
                     'exist').format(os.path.dirname(log_file_path)))
    return
  logging.info('Saving periodic checkpoint to {}'.format(log_file_path))

  save_checkpoint(log_file_path, conf, runner, epoch, best_val_metrics)

  num_checkpoints = conf.get_attr('num_periodic_checkpoints',
                                  default=DEFAULT_NUM_PERIODIC_CHECKPOINTS)
  prune_checkpoints(os.path.dirname(log_file_path), num_checkpoints)


def save_best_checkpoint(best_dir, best_val, conf, runner,
                         epoch, best_val_metrics):
  log_file_path = get_best_checkpoint_path(best_dir, epoch, best_val)
  if not os.path.isdir(os.path.dirname(log_file_path)):
    logging.warning(('Skip saving best value checkpoint: {} does not '
                     'exist').format(os.path.dirname(log_file_path)))
    return

  logging.info('Saving best value checkpoint to {}'.format(log_file_path))

  save_checkpoint(log_file_path, conf, runner,
                  epoch, best_val_metrics)

  num_checkpoints = conf.get_attr('num_best_checkpoints',
                                  default=DEFAULT_NUM_BEST_CHECKPOINTS)
  prune_checkpoints(os.path.dirname(log_file_path), num_checkpoints)


def make_comparison_grid(targets, predictions, num_images, **kwargs):
  if isinstance(targets, Variable):
    targets = targets.data
  if isinstance(predictions, Variable):
    predictions = predictions.data

  images = []
  for idx, (target, prediction) in enumerate(zip(targets, predictions)):
    if idx >= num_images:
      break

    images += [target, prediction]

  nrows = int(math.ceil(len(images) / 4))
  return make_grid(images, nrow=nrows, **kwargs)


def save_images_to_tensorboard(summary_writer, conf, num_image_summaries,
                               global_step, tag, prediction, target,
                               task_name):
  def convert_to_segmentations(prediction, target):
    tasks = conf.get_attr('tensorboard_segmentation_tasks', default=None)
    if tasks is not None and task_name not in tasks:
      return prediction, target

    # Convert segmentations to gray values to visualize them
    num_classes = conf.get_attr('num_classes', default=None)
    if num_classes is None:
      max_idx = max(prediction.max()[0].data[0], target.max()[0].data[0])
      num_classes = max(2, max_idx + 1)
    prediction = prediction.type(torch.FloatTensor) / (num_classes - 1)
    target = target.type(torch.FloatTensor) / (num_classes - 1)
    return prediction, target

  if prediction.shape != target.shape:
    logging.warning(('Shape of prediction {} differs from shape '
                     'of target {} while saving images to '
                     'Tensorboard').format(prediction.shape, target.shape))

  if target.size()[0] <= num_image_summaries:
    num_images = target.size()[0]
  else:
    num_images = num_image_summaries

  if target.dim() != 4 or (target.shape[1] != 1 and target.shape[1] != 3):
    # No image data to visualize
    logging.debug(('Skipping writing images with shape {} and tag {} to '
                   'Tensorboard').format(target.shape, tag))
    return num_images

  if conf.get_attr('tensorboard_segmentation', default=False):
    prediction, target = convert_to_segmentations(prediction, target)

  grid = make_comparison_grid(target, prediction, num_images)
  summary_writer.add_image(tag, grid, global_step)
  return num_images


def run_validation(conf, runner, epoch, val_loader, best_val_metrics,
                   chkpt_metric_dirs, summary_writer, num_batches_per_epoch,
                   early_stoppers):
  best_value_warmup = conf.get_attr('best_value_warmup_epochs',
                                    default=DEFAULT_BEST_VALUE_WARMUP_EPOCHS)
  num_image_summaries = conf.get_attr('num_image_summaries',
                                      default=DEFAULT_NUM_IMAGE_SUMMARIES)
  num_batches = np.ceil(num_image_summaries / val_loader.batch_size)

  val_start_time = time.time()
  res = runner.validate(val_loader, num_batches_to_return=num_batches)
  data, val_losses, val_metrics = res
  val_duration = time.time() - val_start_time

  s = '===> Validation: '
  s += ', '.join(('{}: {}'.format(name, loss)
                  for name, loss in val_losses.items()))
  s += ', time: {:.4f}s\n'.format(val_duration)
  s += '\n'.join(('     {}: {}'.format(name, metric)
                  for name, metric in val_metrics.items()))
  logging.info(s)

  for name, value in chain(val_losses.items(), val_metrics.items()):
    if epoch <= best_value_warmup:
      continue

    best_value = False
    if name in best_val_metrics:
      if value > best_val_metrics[name]:
        best_val_metrics[name] = value
        best_value = True
    else:
      best_val_metrics[name] = value
      best_value = True

    if best_value and name in chkpt_metric_dirs:
      save_best_checkpoint(chkpt_metric_dirs[name], value.value,
                           conf, runner, epoch + 1, best_val_metrics)

    for early_stopper in early_stoppers:
      if name == early_stopper.name:
        if best_value:
          early_stopper.record_best_value(value, epoch)
        early_stopper.record_value(value, epoch)

  if summary_writer is not None:
    global_step = num_batches_per_epoch * epoch
    for metric_name, metric in chain(val_losses.items(), val_metrics.items()):
      summary_writer.add_scalar('validation/{}'.format(metric_name),
                                metric.value, global_step)

    if num_image_summaries > 0:
      for idx, batch in enumerate(data):
        named_batch = runner.get_named_outputs(batch)

        if 'outputs_by_task' in named_batch:
          outputs_by_task = named_batch['outputs_by_task']
          for task_name, (prediction, target) in outputs_by_task.items():
            tag = 'validation/targets_and_predictions_{}_{}'.format(task_name,
                                                                    idx)
            num_images = save_images_to_tensorboard(summary_writer, conf,
                                                    num_image_summaries,
                                                    global_step, tag,
                                                    prediction, target,
                                                    task_name)

        if 'prediction' in named_batch:
          prediction = named_batch['prediction']
          target = named_batch['target']
          tag = 'validation/targets_and_predictions_{}'.format(idx)
          num_images = save_images_to_tensorboard(summary_writer, conf,
                                                  num_image_summaries,
                                                  global_step, tag,
                                                  prediction, target,
                                                  task_name='default')

        num_image_summaries -= num_images
        if num_image_summaries <= 0:
          break


def train_net(conf, runner, train_loader, val_loader, cuda,
              chkpt_metric_dirs={}, restore_state=None, summary_writer=None,
              early_stoppers=[]):
  num_batches_per_epoch = len(train_loader)
  epochs_per_checkpoint = conf.get_attr('epochs_per_checkpoint',
                                        default=DEFAULT_EPOCHS_PER_CHECKPOINT)
  epochs_per_validation = conf.get_attr('epochs_per_validation',
                                        default=DEFAULT_EPOCHS_PER_VALIDATION)
  steps_per_summary = conf.get_attr('steps_per_train_summary',
                                    default=DEFAULT_STEPS_PER_TRAIN_SUMMARY)
  initial_validation = conf.get_attr('initial_validation',
                                     default=DEFAULT_INITIAL_VALIDATION)

  if restore_state is None:
    start_epoch = 1
    best_val_metrics = {}
  else:
    assert 'start_epoch' in restore_state \
        and 'best_val_metrics' in restore_state, \
        'Invalid checkpoint for resuming training. Inference checkpoint?'
    start_epoch = restore_state['start_epoch']
    best_val_metrics = restore_state['best_val_metrics']

  if restore_state is None and initial_validation:
    initial_epoch = 0
    logging.info('Running pretraining validation')
    run_validation(conf, runner, initial_epoch, val_loader, best_val_metrics,
                   chkpt_metric_dirs, summary_writer, num_batches_per_epoch)
    save_periodic_checkpoint(conf, runner, initial_epoch, best_val_metrics)

  for epoch in range(start_epoch, conf.num_epochs + 1):
    runner.epoch_beginning(epoch)

    epoch_start_time = time.time()
    train_losses, train_metrics = runner.train_epoch(train_loader,
                                                     epoch,
                                                     summary_writer,
                                                     steps_per_summary,
                                                     conf.args.verbose)
    epoch_duration = time.time() - epoch_start_time

    runner.epoch_finished(epoch)

    s = '===> Epoch {} Complete: '.format(epoch)
    s += ', '.join(('{}: {}'.format(name, loss)
                    for name, loss in train_losses.items()))
    s += ', time: {:.4f}s\n'.format(epoch_duration)
    s += '\n'.join(('     {}: {}'.format(name, metric)
                    for name, metric in train_metrics.items()))
    logging.info(s)

    if epoch % epochs_per_validation == 0:
      run_validation(conf, runner, epoch, val_loader, best_val_metrics,
                     chkpt_metric_dirs, summary_writer, num_batches_per_epoch,
                     early_stoppers)

    if epoch % epochs_per_checkpoint == 0 or epoch == conf.num_epochs:
      save_periodic_checkpoint(conf, runner, epoch + 1, best_val_metrics)

    early_stopping_triggered = False
    for early_stopper in early_stoppers:
      if early_stopper.should_stop(epoch):
        logging.info(early_stopper.stop_reason(epoch))
        early_stopping_triggered = True
        break

    if early_stopping_triggered:
      break


def main(argv):
  args = parser.parse_args(argv)

  # Load configuration
  conf = Configuration.from_json(args.config)
  conf.args = args
  if args.conf:
    new_conf_entries = {}
    for arg in args.conf:
      key, value = arg.split('=')
      new_conf_entries[key] = value
    conf.update(new_conf_entries)

  # Setup log directory
  if args.run_dir:
    conf.run_dir = args.run_dir
  elif args.resume:
    if os.path.exists(args.resume):
      conf.run_dir = os.path.dirname(args.resume)
  if not conf.has_attr('run_dir'):
    run_name = conf.get_attr('run_name', default='unnamed_run')
    conf.run_dir = get_run_dir(args.log_dir, run_name)
  if not args.dry:
    if not os.path.isdir(conf.run_dir):
      os.mkdir(conf.run_dir)

  setup_logging(conf.run_dir, 'train', args.verbose, args.dry)

  logging.info('Commandline arguments: {}'.format(' '.join(argv)))

  if not args.dry:
    logging.info('This run is saved to: {}'.format(conf.run_dir))
    config_path = get_config_path(conf.run_dir)
    conf.serialize(config_path)

  if args.cuda != '':
    try:
      args.cuda = utils.set_cuda_env(args.cuda)
    except Exception:
      logging.critical('No free GPU on this machine. Aborting run.')
      return
    logging.info('Running on GPU {}'.format(args.cuda))

  if args.verbose:
    logging.debug(str(conf))

  utils.set_random_seeds(conf.seed)

  # Setup model
  logging.info('Setting up training runner {}'.format(conf.runner_type))
  runner = build_runner(conf, conf.runner_type, args.cuda, mode='train')

  if args.print_model:
    print(str(runner))

  if args.print_parameters:
    print_model_parameters(runner)

  # Handle resuming from checkpoint
  restore_state = None
  if args.resume:
    if os.path.exists(args.resume):
      restore_state = restore_checkpoint(args.resume, runner)
      logging.info('Restored checkpoint from {}'.format(args.resume))
    else:
      logging.critical(('Checkpoint {} to restore '
                        'from not found').format(args.resume))
      return

  use_tensorboard = conf.get_attr('use_tensorboard',
                                  default=DEFAULT_USE_TENSORBOARD)
  if use_tensorboard and not args.dry:
    from tensorboardX import SummaryWriter
    summary_writer = SummaryWriter(conf.run_dir)
    logging.debug('Using tensorboardX summary writer')
  else:
    summary_writer = None

  # Load datasets
  num_workers = conf.get_attr('num_data_workers', default=DEFAULT_NUM_WORKERS)
  num_train_samples = conf.get_attr('num_train_subset_samples', default=None)
  num_val_samples = conf.get_attr('num_validation_subset_samples',
                                  default=None)

  train_dataset_name = conf.get_attr('train_dataset', alternative='dataset')
  logging.info('Loading training dataset {}'.format(train_dataset_name))
  train_dataset = load_dataset(conf, args.data_dir,
                               train_dataset_name, 'train')
  train_sampler = maybe_get_subset_sampler(num_train_samples, train_dataset)
  train_loader = DataLoader(dataset=train_dataset,
                            num_workers=num_workers,
                            batch_size=conf.batch_size,
                            sampler=train_sampler,
                            shuffle=train_sampler is None,
                            worker_init_fn=utils.set_worker_seeds)

  val_dataset_name = conf.get_attr('validation_dataset', alternative='dataset')
  logging.info('Loading validation dataset {}'.format(val_dataset_name))
  val_dataset = load_dataset(conf, args.data_dir, val_dataset_name, 'val')
  val_sampler = maybe_get_subset_sampler(num_val_samples, val_dataset)
  val_loader = DataLoader(dataset=val_dataset,
                          num_workers=num_workers,
                          batch_size=conf.get_attr('validation_batch_size',
                                                   default=conf.batch_size),
                          sampler=val_sampler,
                          shuffle=False,
                          worker_init_fn=utils.set_worker_seeds)

  # Setup validation checkpoints
  chkpt_metrics = conf.get_attr('validation_checkpoint_metrics', default=[])
  chkpt_metric_dirs = {metric: os.path.join(conf.run_dir, 'best_' + metric)
                       for metric in chkpt_metrics}
  for metric_dir in chkpt_metric_dirs.values():
    if not args.dry and not os.path.isdir(metric_dir):
      os.mkdir(metric_dir)

  # Setup early stopping
  if conf.has_attr('early_stopping'):
    from training.early_stopping import EarlyStopper
    early_stoppers = [EarlyStopper(conf.early_stopping['metric_name'],
                                   conf.early_stopping['patience'],
                                   conf.early_stopping.get('min_value',
                                                           None),
                                   conf.early_stopping.get('max_difference',
                                                           None))]
  elif conf.has_attr('early_stoppers'):
    from training.early_stopping import EarlyStopper
    early_stoppers = []
    for early_stopping_conf in conf.early_stoppers:
      min_value = early_stopping_conf.get('min_value', None)
      max_diff = early_stopping_conf.get('max_difference', None)
      early_stoppers.append(EarlyStopper(early_stopping_conf['metric_name'],
                                         early_stopping_conf['patience'],
                                         min_value, max_diff))
  else:
    early_stoppers = []

  logging.info('Starting training run of {} epochs'.format(conf.num_epochs))

  # Train
  try:
    train_net(conf, runner, train_loader, val_loader, args.cuda,
              chkpt_metric_dirs, restore_state, summary_writer, early_stoppers)
  except KeyboardInterrupt:
    if summary_writer is not None:
      summary_writer.close()


if __name__ == '__main__':
  main(sys.argv[1:])
