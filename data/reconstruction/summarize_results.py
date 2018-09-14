#!/usr/bin/env python
"""Analyze metrics stored in panda tables"""
import argparse
import os
import sys
import re
from collections import OrderedDict
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

NAME_REGEXP = re.compile(r'.+_(.+)_\d\d\d\d.+')
SIGNIFICANCE_LVL = 0.05

REC_DICE_GT = 0.7964832518779061

parser = argparse.ArgumentParser(description='Evaluate metrics')
parser.add_argument('-v', action='store_true', help='Verbosity')
parser.add_argument('-o', '--order', help='Output order')
parser.add_argument('-p', default='auto', help='Floating-point precision')
parser.add_argument('-l', action='store_true', help='Output latex markup')
parser.add_argument('-f', '--filter',
                    help='Filter outputs by substring')
parser.add_argument('--sis-gt-perf', default=REC_DICE_GT,
                    help='Performance on GT for SIS')
parser.add_argument('--pprint', action='store_true',
                    help='Print out percentiles')
parser.add_argument('--percentiles', default=[0, 25, 50, 75, 100],
                    help='Percentiles to print')
parser.add_argument('--stest', action='store_true',
                    help='Perform statistical testing')
parser.add_argument('--sprint', action='store_true',
                    help='Print results of statistical testing')
parser.add_argument('--slvl', default=SIGNIFICANCE_LVL,
                    help='Significance level')
parser.add_argument('--stest-mode', default='wilcoxon',
                    choices=('ttest', 'wilcoxon'),
                    help='Mode of statistical testing')
parser.add_argument('--no-name', action='store_true',
                    help='Do not print leading run name')
parser.add_argument('--no-std', action='store_true',
                    help='Do not print std')
parser.add_argument('--metric-name', default='dice_avg',
                    help='Metric name to aggregate')
parser.add_argument('inputs', nargs='+', help='Input csvs to process')


def get_best_fn(metric_name):
  max_metrics = ['dice', 'psnr', 'ssim', 'segscore']
  for metric in max_metrics:
    if metric in metric_name.lower():
      return max
  return min


def get_precision(metric_name):
  default = 2
  precisions = {
      'dice': 3,
      'segscore': 3,
      'ssim': 3
  }
  for metric, prec in precisions.items():
    if metric in metric_name:
      return prec
  return default


def statistical_testing(args, metrics_by_input, groups_by_name):
  test_fn = ttest_rel if args.stest_mode == 'ttest' else wilcoxon

  # Get group averages
  samples_by_name = {}
  for name, group in groups_by_name.items():
    gmeans = np.mean([metrics_by_input[inp] for inp in group], axis=0)
    samples_by_name[name] = gmeans

  perms = permutations(samples_by_name.items(), 2)

  if args.sprint:
    print('Performing {}'.format(args.stest_mode))
  tested_names = set()
  pvalues_by_name = {}
  for (n1, s1), (n2, s2) in perms:
    if n1 not in tested_names:
      if args.sprint:
        print('Testing {} against:'.format(n1))
      tested_names.add(n1)

    assert len(s1) == len(s2)

    test = test_fn(s1, s2)
    pvalues_by_name.setdefault(n1, []).append(test.pvalue)
    if args.sprint:
      print('\t{}: {:.4f}'.format(n2, test.pvalue))

  significantly_different_names = []
  for name, pvalues in pvalues_by_name.items():
    if all((p < args.slvl) for p in pvalues):
      significantly_different_names.append(name)
      if args.sprint:
        print(('{} ({:.3f}) has p < {} '
               'for all other inputs').format(name,
                                              samples_by_name[name].mean(),
                                              args.slvl))

  return significantly_different_names


def collect_mean_std(args, metric_name, metrics_by_input, groups_by_name):
  gavgs_by_name = OrderedDict()
  for name, group in groups_by_name.items():
    gmean = np.mean([metrics_by_input[inp].mean() for inp in group])
    gstd = np.mean([metrics_by_input[inp].std() for inp in group])
    gavgs_by_name[name] = (gmean, gstd)
    if args.v:
      means = [metrics_by_input[inp].mean() for inp in group]
      print(name, ','.join(('{:.3f}'.format(m) for m in means)),
            '({:.3f} +- {:.3f})'.format(gmean, np.std(means)))

  if 'segscore' in metric_name.lower():
    for name, gavg in gavgs_by_name.items():
      gavgs_by_name[name] = (gavg[0] / args.sis_gt_perf, 0)

  return gavgs_by_name


def print_mean_std(args, metric_name, gavgs_by_name,
                   significantly_different_names, name_order):
  best_fn = get_best_fn(metric_name)
  best_val = best_fn(gavgs_by_name, key=lambda k: gavgs_by_name[k])

  if args.p == 'auto':
    prec = get_precision(metric_name)
  else:
    prec = args.p

  max_width = max((len(inp) for inp in gavgs_by_name))
  str_fmt = '{:' + str(max_width+2) + '}'
  fp_fmt = '{:.' + str(prec) + 'f}'

  if len(name_order) == 2:
    name_order.append('diff')
    mdiff = gavgs_by_name[name_order[1]][0] - gavgs_by_name[name_order[0]][0]
    sdiff = gavgs_by_name[name_order[1]][1] - gavgs_by_name[name_order[0]][1]
    gavgs_by_name['diff'] = (mdiff, sdiff)

  for name in name_order:
    (mean, std) = gavgs_by_name[name]
    s = ''
    mean_fmt = fp_fmt
    std_fmt = fp_fmt
    delim = ' '
    mean_std_delim = ' +- '

    if args.l:
      delim = '$'
      mean_std_delim = ' \pm '
      if args.stest and name in significantly_different_names:
        mean_fmt += '^{{*}}'

      if name == best_val:
        mean_fmt = '\mathbf{{' + mean_fmt + '}}'
    else:
      if args.stest and name in significantly_different_names:
        mean_fmt += '*'

    if not args.no_name:
      s += str_fmt.format(name)

    s += delim + mean_fmt.format(mean)

    if not args.no_std:
      s += mean_std_delim + std_fmt.format(std)
    s += delim

    print(s)


def print_percentiles(args, metric_name, metrics_by_input, groups_by_name, name_order):
  if args.p == 'auto':
    prec = 3 if 'dice' in metric_name else 2
  else:
    prec = args.p

  # Get group averages
  samples_by_name = {}
  for name, group in groups_by_name.items():
    gmeans = np.mean([metrics_by_input[inp] for inp in group], axis=0)
    samples_by_name[name] = gmeans

  max_width = max((len(name) for name in groups_by_name))
  str_fmt = '{:' + str(max_width+2) + '}'
  fp_fmt = '{:.' + str(prec) + 'f}'

  percs_by_name = {name: np.percentile(samples_by_name[name],
                                       args.percentiles)
                   for name in name_order}
  if len(name_order) == 2:
    name_order.append('diff')
    pdiff = percs_by_name[name_order[1]] - percs_by_name[name_order[0]]
    percs_by_name['diff'] = pdiff

  for name in name_order:
    percs = percs_by_name[name]
    s = ''
    if not args.no_name:
      s += str_fmt.format(name)

    if args.l:
      s += '$'
      s += '/'.join((fp_fmt.format(p) for p in percs))
      s += '$'
    else:
      s += '/'.join((fp_fmt.format(p) for p in percs))

    print(s)


def evaluate_for_metric(args, dfs, metric_name):
  metrics_by_input = {}
  for name, df in dfs.items():
    df = df.dropna(subset=[metric_name])
    metrics_by_input[name] = df[metric_name]

  if args.v:
    print('Available columns in {}'.format(inp))
    print(list(df.columns))

  groups_by_name = OrderedDict()
  for inp in metrics_by_input:
    m = NAME_REGEXP.match(inp)
    assert m is not None, inp
    groups_by_name.setdefault(m.group(1), []).append(inp)

  if args.filter is not None:
    filtered_groups_by_name = OrderedDict()
    for name in groups_by_name:
      if not any((name_to_filter in name for name_to_filter in args.filter)):
          filtered_groups_by_name[name] = groups_by_name[name]
    groups_by_name = filtered_groups_by_name

  if args.order is not None:
      name_order = []
      for key in args.order:
        for name in groups_by_name:
          if key in name and name not in name_order:
            name_order.append(name)
            break
  else:
    name_order = list(groups_by_name.keys())

  if args.pprint:
    print_percentiles(args, metric_name, metrics_by_input, groups_by_name,
                      name_order)
  elif not args.sprint:
    gavgs_by_name = collect_mean_std(args, metric_name,
                                     metrics_by_input, groups_by_name)
    significantly_different_names = statistical_testing(args,
                                                        metrics_by_input,
                                                        groups_by_name)
    print_mean_std(args, metric_name, gavgs_by_name,
                   significantly_different_names, name_order)
  else:
    statistical_testing(args, metrics_by_input, groups_by_name)


def main(argv):
  args = parser.parse_args(argv)

  if args.order is not None:
    args.order = args.order.split(',')
  if args.filter is not None:
    args.filter = args.filter.split(',')

  args.inputs = [inp for inp in args.inputs if inp.endswith('.csv')]

  dfs = {}
  for inp in args.inputs:
    df = pd.read_csv(inp)
    name = os.path.basename(inp)
    dfs[name] = df

  metric_names = args.metric_name.split(',')
  for metric_name in metric_names:
    print(metric_name)
    evaluate_for_metric(args, dfs, metric_name)
    print()

if __name__ == '__main__':
  main(sys.argv[1:])
