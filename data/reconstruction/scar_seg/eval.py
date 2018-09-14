#!/usr/bin/env python
"""Evaluate and write out metrics"""
import argparse
import os
import sys
from functools import partial

import numpy as np
import pandas as pd

from data.reconstruction.io import CaseDataset, CASE_KEY
from data.reconstruction.scar_seg.io import (load_dataset,
                                             add_gt_labels)
from data.reconstruction.statistics import (compute_psnr,
                                            compute_ssim,
                                            compute_seg_score)
from utils.config import Configuration

CLASS_IDX = 1
DEFAULT_SEGSCORE_CONF = 'configs/segscore_unet.json'
DEFAULT_SEGSCORE_CONF_RELDIR = 'configs'

parser = argparse.ArgumentParser(description='Evaluate metrics')
parser.add_argument('--src-path', default='resources/data/scar_segmentation',
                    help='Path to dataset folder')
parser.add_argument('--results-path', default='resources/models/results',
                    help='Path to results folder')
parser.add_argument('--fold', default='test',
                    help='Fold')
parser.add_argument('--segscore-conf', default=DEFAULT_SEGSCORE_CONF,
                    help='SegScore config to use')
parser.add_argument('--segscore-conf-reldir',
                    default=DEFAULT_SEGSCORE_CONF_RELDIR,
                    help='Folder to which pretrained path is relative to')
parser.add_argument('-c', '--cuda', default='',
                    help='CUDA value')
parser.add_argument('input', help='Path to reconstructed images')


def load(dataset_path, scar_seg_path, fold):
  dataset = load_dataset(dataset_path)
  add_gt_labels(dataset, scar_seg_path, fold=fold)
  return CaseDataset(dataset)


def get_seg_score_obj(dataset, conf_path, cuda, conf_rel_path):
  from metrics.segmentation_score import SegmentationScore
  conf = Configuration.from_json(conf_path)
  seg_score = SegmentationScore(conf, conf_rel_path, cuda,
                                class_idx=CLASS_IDX, skip_empty_images=True)
  return seg_score


def main(argv):
  # Run from main directory with python -m data.reconstruction.scar_seg.eval
  args = parser.parse_args(argv)

  if args.cuda != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

  dataset = load(args.input, args.src_path, args.fold)

  seg_score = get_seg_score_obj(dataset, args.segscore_conf, args.cuda,
                                args.segscore_conf_reldir)

  metrics = [('PSNR', 'psnr', compute_psnr),
             ('SSIM', 'ssim', compute_ssim)]
             ('SegScore', 'segscore', partial(compute_seg_score,
                                              seg_score=seg_score))]

  out_str = ''
  metric_dfs = []
  for name, metric_id, metric_fn in metrics:
    df = metric_fn(dataset).to_frame(metric_id)
    metric_dfs.append(df)

    s = '{}:\n'.format(name)
    s += '{:.4f} +- {:.4f}\n'.format(np.mean(df[metric_id]),
                                     np.std(df[metric_id]))
    print(s)
    out_str += s + '\n'

  names = pd.Series([data[CASE_KEY] for data in dataset], name='name')
  df = metric_dfs[0].join([names] + metric_dfs[1:])

  csv_name = 'scarseg_{}.csv'.format(os.path.basename(os.path.normpath(args.input)))
  txt_name = 'scarseg_{}.txt'.format(os.path.basename(os.path.normpath(args.input)))

  df.to_csv(os.path.join(args.results_path, csv_name), sep=str(','))
  with open(os.path.join(args.results_path, txt_name), 'w') as f:
    f.write(out_str)


if __name__ == '__main__':
  main(sys.argv[1:])
