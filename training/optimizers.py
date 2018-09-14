import torch.nn as nn
import torch.optim as optim


def get_optimizer(conf, optimizer_name, variables_or_model):
  if isinstance(variables_or_model, nn.Module):
    variables = variables_or_model.parameters()
    if isinstance(variables, dict):
      assert conf.has_attr('parameter_key'), \
          ('Parameter key unspecfied, but model requires one. '
           'Possible keys: {}').format(', '.join(variables.keys()))
      variables = variables[conf.parameter_key]
  else:
    variables = variables_or_model

  if optimizer_name == 'RMSProp':
    alpha = conf.get_attr('alpha', default=0.99)
    return optim.RMSprop(variables, conf.learning_rate, alpha=alpha)
  elif optimizer_name == 'Adam':
    beta1 = conf.get_attr('beta1', default=0.9)
    beta2 = conf.get_attr('beta2', default=0.999)
    return optim.Adam(variables, conf.learning_rate, betas=(beta1, beta2))
  else:
    raise ValueError('Unknown optimizer {}'.format(optimizer_name))
