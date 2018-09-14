import random

import torch
from torch.autograd import Variable


class ImagePool():
  """Image pool keeping a history of images

  Useful for adversarial training to force discriminator to not forget
  previously generated images
  """
  def __init__(self, pool_size, p_pool_image=0.5):
    """Build the pool

    Parameters
    ----------
    pool_size : int
      Number of images the pool can maximally contain. If zero,
      pool is disabled
    p_pool_image : float
      Probability to sample from image pool instead of keeping image from
      image batch
    """
    self.pool_size = pool_size
    self.p_pool_image = p_pool_image
    self.images = []

  def query(self, image_batch):
    """Sample image batch from the pool mixed with images from an input batch

    Parameters
    ----------
    image_batch : torch.Tensor
      Input image batch to use. If sampling from pool, substitutes image from
      pool with image from input image batch
    """
    if self.pool_size == 0:
        return image_batch

    result_batch = []
    for image in image_batch.data:
      image = torch.unsqueeze(image, 0)

      if len(self.images) < self.pool_size:
        # Pool is not yet full, fill it up
        self.images.append(image)
        result_batch.append(image)
      else:
        p = random.uniform(0, 1)
        if p < self.p_pool_image:
          # Sample from pool
          idx = random.randint(0, self.pool_size - 1)
          result_batch.append(self.images[idx].clone())
          self.images[idx] = image
        else:
          # Keep image from input batch
          result_batch.append(image)

    return Variable(torch.cat(result_batch, 0))
