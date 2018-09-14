#!/usr/bin/env python
import argparse
import logging
import os
import sys

from torch.utils.data import DataLoader
from torchvision.utils import save_image

import utils
from data import load_dataset, is_dataset, maybe_get_subset_sampler
from training import build_runner
from utils.checkpoints import restore_checkpoint
from utils.checkpoint_paths import get_run_dir
from utils.config import Configuration
from utils.logging import setup_logging

DEFAULT_NUM_WORKERS = 1

parser = argparse.ArgumentParser(description=('Validate model and infer'
                                              ' predictions on images'))
parser.add_argument('-c', '--cuda', default='0', type=str, help='GPU to use')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print more info')
parser.add_argument('--dry', action='store_true',
                    help=('Do not create output directories. '
                          'Useful for debugging'))
parser.add_argument('--data-dir', default='resources/data',
                    help='Path to data directory')
parser.add_argument('--out-dir', default='resources/outputs',
                    help='Path to where to save outputs')
parser.add_argument('-i', '--infer', action='store_true',
                    help='Save predicted images')
parser.add_argument('-d', '--dump', action='store_true',
                    help='Save input, target and predicted images')
parser.add_argument('--raw', action='store_true',
                    help='Save network outputs in matrix format')
parser.add_argument('-f', '--fold', choices=['train', 'val', 'test'],
                    default='val', help='Fold of dataset to use')
parser.add_argument('--conf', nargs='+',
                    help='Optional config values to set')
parser.add_argument('config', help='Config file to use')
parser.add_argument('checkpoint', help='Checkpoint to use weights from')
parser.add_argument('files_or_dirs', nargs='*',
                    help='Files or folders to evaluate')


def _save_image(image, path):
  if image.shape[0] == 2:
    # Complex image, convert to magnitude image
    image = (image[0] ** 2 + image[1] ** 2) ** 0.5

  save_image(image, path)


def save_output_images(dataset, inputs, predictions, targets, output_dir,
                       filenames, task_name, dump, raw):
  for name, inp, prediction, target in zip(filenames, inputs,
                                           predictions, targets):
    if task_name == 'default':
      prefix = name
    else:
      prefix = '{}_{}'.format(name, task_name)

    if raw:
      assert inputs.shape[0] == 1
      # For now, hardcode to reconstruction case
      from data.reconstruction.io import save_raw
      filepath = os.path.join(output_dir, '{}.mat'.format(prefix))
      save_raw(filepath, name,
               inputs.data.numpy(),
               predictions.data.numpy(),
               targets.data.numpy(),
               dataset)
    else:
      # Try to save as images
      if dump:
        input_file = os.path.join(output_dir, '{}_input.png'.format(prefix))
        _save_image(inp.data, input_file)
        target_file = os.path.join(output_dir, '{}_target.png'.format(prefix))
        _save_image(target.data, target_file)
      pred_file = os.path.join(output_dir, '{}_pred.png'.format(prefix))
      _save_image(prediction.data, pred_file)
      logging.debug('Wrote images for {}, task {}'.format(name, task_name))


def main(argv):
  args = parser.parse_args(argv)

  setup_logging(os.path.dirname(args.checkpoint), 'eval',
                args.verbose, args.dry)

  logging.info('Commandline arguments: {}'.format(' '.join(argv)))

  if args.cuda != '':
    try:
      args.cuda = utils.set_cuda_env(args.cuda)
    except Exception:
      logging.critical('No free GPU on this machine. Aborting run.')
      return
    logging.info('Running on GPU {}'.format(args.cuda))

  # Load configuration
  conf = Configuration.from_json(args.config)
  conf.args = args
  if args.conf:
    new_conf_entries = {}
    for arg in args.conf:
      key, value = arg.split('=')
      new_conf_entries[key] = value
    conf.update(new_conf_entries)

  if args.verbose:
    logging.debug(conf)

  utils.set_random_seeds(conf.seed)

  if args.raw:
    # This is a hack to suppress the output transform when we request raw data
    conf.application = 'none'
    if conf.has_attr('tasks'):
      for name, task in conf.tasks.items():
        if 'application' in task:
          logging.debug(('Changing output transform in task {} '
                         'from {} to none').format(name,
                                                   task['application']))
          task['application'] = 'none'

  # Setup model
  runner = build_runner(conf, conf.runner_type, args.cuda, mode='test')

  # Handle resuming from checkpoint
  if args.checkpoint != 'NONE':
    if os.path.exists(args.checkpoint):
      _ = restore_checkpoint(args.checkpoint, runner, cuda=args.cuda)
      logging.info('Restored checkpoint from {}'.format(args.checkpoint))
    else:
      logging.critical(('Checkpoint {} to restore '
                       'from not found').format(args.checkpoint))
      return

  # Load datasets
  mode = 'dataset'
  if len(args.files_or_dirs) == 0:
    datasets = [load_dataset(conf, args.data_dir,
                             conf.validation_dataset, args.fold)]
  else:
    datasets = []
    for f in args.files_or_dirs:
      if is_dataset(f):
        dataset = load_dataset(conf, args.data_dir, f, args.fold)
        datasets.append(dataset)

  if args.raw:
    mode = 'raw'

  num_samples = conf.get_attr('num_validation_subset_samples',
                              default=None)

  # Evaluate all datasets
  for dataset in datasets:
    logging.info('Evaluating dataset {}'.format(dataset.name))

    sampler = maybe_get_subset_sampler(num_samples, dataset)
    loader = DataLoader(dataset=dataset,
                        num_workers=DEFAULT_NUM_WORKERS,
                        batch_size=1,
                        sampler=sampler,
                        shuffle=False)

    if mode == 'dataset':
      data, _, val_metrics = runner.validate(loader, len(loader))

      res_str = 'Average metrics for {}\n'.format(dataset.name)
      for metric_name, metric in val_metrics.items():
        res_str += '     {}: {}\n'.format(metric_name, metric)
      logging.info(res_str)
    else:
      data = runner.infer(loader)

    if not args.dry and (args.infer or args.dump):
      if mode == 'dataset' or mode == 'raw':
        conf_name = os.path.splitext(os.path.basename(conf.file))[0]
        output_dir = get_run_dir(args.out_dir, '{}_{}'.format(dataset.name,
                                                              conf_name))
        if not os.path.isdir(output_dir):
          os.mkdir(output_dir)

      logging.info('Writing images to {}'.format(output_dir))

      file_idx = 0
      for batch in data:
        if mode == 'image':
          output_dir = os.path.dirname(dataset.images[file_idx])

        named_batch = runner.get_named_outputs(batch)
        inp = named_batch['input']

        if 'prediction' in named_batch:
          batch_size = named_batch['prediction'].shape[0]
          filenames = [dataset.get_filename(idx)
                       for idx in range(file_idx, file_idx + batch_size)]
          save_output_images(dataset, inp, named_batch['prediction'],
                             named_batch['target'], output_dir,
                             filenames, 'default', args.dump, args.raw)

        file_idx += len(filenames)

      logging.info(('Finished writing images for '
                   'dataset {}').format(dataset.name))


if __name__ == '__main__':
  main(sys.argv[1:])
