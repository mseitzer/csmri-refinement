import importlib

RUNNER_MODULES = {
    'standard': 'training.runner',
    'adversarial': 'training.adversarial_runner',
    'multitask': 'training.multitask_runner',
    'adversarial-multitask': 'training.adversarial_multitask_runner'
}


def build_runner(conf, runner_type, cuda, mode='train'):
  assert runner_type in RUNNER_MODULES, \
      'Unknown runner {}'.format(runner_type)

  module = importlib.import_module(RUNNER_MODULES[runner_type])
  runner = module.build_runner(conf, cuda, mode)
  return runner
