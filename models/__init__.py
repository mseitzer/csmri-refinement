import importlib

MODEL_MODULES = {
    'UNET': 'models.unet',
    'CNNDiscriminator': 'models.discriminators',
    'RecNet': 'models.recnet',
    'RefinementWrapper': 'models.refinement_wrapper',
}


def construct_model(conf, model_name, cuda=None):
  assert model_name in MODEL_MODULES, \
      'Unknown model {}'.format(model_name)

  module = importlib.import_module(MODEL_MODULES[model_name])
  model = module.construct_model(conf, model_name, cuda=cuda)
  return model
