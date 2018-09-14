import torch

def _make_byte_tensor(value, shape, cuda):
  if cuda != '':
    return torch.cuda.ByteTensor(shape).fill_(value)
  else:
    return torch.ByteTensor(shape).fill_(value)


def binary_accuracy(prediction, target):
  """Calculates accuracy between two classes using probabilities

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Vector of probabilities of class 1
  target : torch.Tensor
    Vector containing the class indices 0 or 1
  """
  if isinstance(prediction, torch.autograd.Variable):
      prediction = prediction.data
  predicted_classes = torch.gt(prediction, 0.5)
  return torch.mean(torch.eq(predicted_classes, target.byte()).float())


def disc_accuracy(prob_fake, prob_real, fake_accuracy, real_accuracy, cuda):
  """Wrapper for binary accuracy for discriminator"""
  batch_size = prob_fake.shape[0]
  if fake_accuracy:
    if prob_fake.dim() >= 1:
      prob_fake = prob_fake.view(batch_size, -1).mean(dim=1)
    target_fake = _make_byte_tensor(0, prob_fake.shape, cuda)
  if real_accuracy:
    if prob_real.dim() >= 1:
      prob_real = prob_real.view(batch_size, -1).mean(dim=1)
    target_real = _make_byte_tensor(1, prob_real.shape, cuda)

  if fake_accuracy and real_accuracy:
    prob = torch.cat((prob_fake, prob_real), dim=0)
    target = torch.cat((target_fake, target_real), dim=0)
  elif fake_accuracy:
    prob = prob_fake
    target = target_fake
  elif real_accuracy:
    prob = prob_real
    target = target_real
  else:
    raise ValueError('fake_accuracy and real_accuracy can not both be false')

  return binary_accuracy(prob, target)
