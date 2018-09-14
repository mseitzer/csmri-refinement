from collections import Iterable


class Metric(object):
  def __init__(self, values):
    if isinstance(values, Iterable):
      self._value = None
      self.sum_values = 0
      self.num_updates = 0
      for value in values:
        self.sum_values += value
        self.num_updates += 1
    else:
      self._value = values
      self.sum_values = values
      self.num_updates = 1

  @property
  def value(self):
    if self._value is None:
      return self.average().value
    else:
      return self._value

  @property
  def worst_value(self):
    """Return worst value

    Returns value which is worse than all other values according to metric
    """
    raise NotImplementedError('Subclasses must override worst_value')

  def __str__(self):
    abs_value = abs(self.value)
    if abs_value >= 1e-4:
      return '{:.4f}'.format(self.value)
    elif abs_value >= 1e-8:
      return '{:.8f}'.format(self.value)
    else:
      return '{:.12f}'.format(self.value)

  def __gt__(self, other):
    """Returns true iff this metric is better than other according to metric"""
    raise NotImplementedError('Subclasses must override __gt__')

  def accumulate(self, metric):
    self._value = metric._value
    self.sum_values += metric.sum_values
    self.num_updates += metric.num_updates

  def average(self):
    return type(self)(self.sum_values / self.num_updates)


class MinMetric(Metric):
  """Metric for which smaller values are better"""
  def __init__(self, values):
    super(MinMetric, self).__init__(values)

  @property
  def worst_value(self):
    """Return worst value"""
    return MinMetric(float('inf'))

  def __gt__(self, other):
    """Returns true iff this metric is better than other according to metric

    For MinMetric, smaller values are better
    """
    return self.value < other.value


class MaxMetric(Metric):
  """Metric for which larger values are better"""
  def __init__(self, values):
    super(MaxMetric, self).__init__(values)

  @property
  def worst_value(self):
    """Return worst value"""
    return MaxMetric(float('-inf'))

  def __gt__(self, other):
    """Returns true iff this metric is better than other according to metric

    For MaxMetric, larger values are better
    """
    return self.value > other.value
