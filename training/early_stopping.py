

class EarlyStopper(object):
  """Early stopping implementation

  Two modes:
  - stop when metric has not improved for some amount of epochs
  - stop when metric has sunken below some value
  - stop when metric has gotten worse for some amount of epochs,
    and overstepped some limit
  """
  def __init__(self, metric_name, patience,
               min_value=None, max_difference=None):
    assert patience >= 1
    assert (min_value is None) or (max_difference is None)
    self.name = metric_name
    self.patience = patience
    self.values_by_epoch = {}
    self.best_value_epoch = 0
    self.min_value = min_value
    self.max_difference = max_difference

  def stop_reason(self, epoch):
    if self.min_value is not None:
      s = ('Early stopping training in epoch {} because metric {} has '
           'sunken below minimum value {} '
           '(best value {} in epoch {}, last '
           'value {})').format(epoch, self.name, self.min_value,
                               self.values_by_epoch[self.best_value_epoch],
                               self.best_value_epoch,
                               self.values_by_epoch[epoch])
    elif self.max_difference is not None:
      s = ('Early stopping training in epoch {} because metric {} has not '
           'improved since {} epochs, and the difference exceeded {} '
           '(best value {} in epoch {}, last '
           'value {})').format(epoch, self.name, self.patience,
                               self.max_difference,
                               self.values_by_epoch[self.best_value_epoch],
                               self.best_value_epoch,
                               self.values_by_epoch[epoch])
    else:
      s = ('Early stopping training in epoch {} because metric {} has not '
           'improved since {} epochs (best value {} in '
           'epoch {})').format(epoch, self.name, self.patience,
                               self.values_by_epoch[self.best_value_epoch],
                               self.best_value_epoch)
    return s

  def should_stop(self, epoch):
    if self.best_value_epoch == 0:
      # Still in best_value_warmup period
      return False

    no_improvement = self.best_value_epoch + self.patience <= epoch
    if self.min_value is not None:
      current_value = self.values_by_epoch[epoch]
      return current_value.value < self.min_value
    elif self.max_difference is not None:
      current_value = self.values_by_epoch[epoch]
      patience_value = self.values_by_epoch[max(epoch - self.patience, 1)]
      diff = abs(current_value.value - patience_value.value)
      # If difference between now and patience epoch is too large, and all
      # values between patience epoch and now are worse than patience epoch,
      # stop training
      return diff > self.max_difference and no_improvement
    else:
      return no_improvement

  def record_best_value(self, best_value, epoch):
    self.best_value_epoch = epoch

  def record_value(self, value, epoch):
    self.values_by_epoch[epoch] = value
