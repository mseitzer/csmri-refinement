import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def compute_psnr(prediction, target):
  """Calculates peak signal-to-noise ratio between target and prediction

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Predicted image
  target : torch.Tensor or torch.autograd.Variable
    Target image
  """
  mse = F.mse_loss(prediction, target).data[0]
  psnr = 10. * np.log10(1. / mse)
  return psnr


def compute_ssim(prediction, target, window_size=11):
  """Calculates structural similarity index between target and prediction

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Predicted image
  target : torch.Tensor or torch.autograd.Variable
    Target image
  window_size : int
    Size of the Gaussian kernel used for computing SSIM
  """
  from metrics import pytorch_ssim

  if not isinstance(prediction, Variable):
    prediction = Variable(prediction, volatile=True)
  if not isinstance(target, Variable):
    target = Variable(target, volatile=True)

  ssim = pytorch_ssim.ssim(prediction, target, window_size=window_size).data[0]
  return ssim


def compute_hfen(prediction, target):
  """Calculates high frequency error norm [1] between target and prediction

  Implementation follows [2], who define a normalized version of HFEN.

  [1]: Ravishankar and Bresler: MR Image Reconstruction From Highly
  Undersampled k-Space Data by Dictionary Learning, 2011
  [2]: Han et al: Image Reconstruction Using Analysis Model Prior, 2016

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Predicted image
  target : torch.Tensor or torch.autograd.Variable
    Target image
  """
  from scipy.ndimage.filters import gaussian_laplace
  # HFEN is defined to use a kernel of size 15x15. Kernel size is defined as
  # 2 * int(truncate * sigma + 0.5) + 1, so we have to use truncate=4.5
  pred_filtered = gaussian_laplace(prediction.data, truncate=4.5, sigma=1.5)
  target_filtered = gaussian_laplace(target.data, truncate=4.5, sigma=1.5)

  norm_diff = np.linalg.norm((pred_filtered - target_filtered).flatten())
  norm_target = np.linalg.norm(target_filtered.flatten())

  return norm_diff / norm_target


def compute_mutual_information(prediction, target):
  """Calculates mutual information between target and prediction

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Predicted image
  target : torch.Tensor or torch.autograd.Variable
    Target image
  """
  from sklearn.metrics import mutual_info_score
  p_xy, _, _ = np.histogram2d(prediction.data.cpu().numpy().flatten(),
                              target.data.cpu().numpy().flatten(),
                              bins=256,
                              range=((0, 1), (0, 1)),
                              normed=True)
  return mutual_info_score(None, None, contingency=p_xy)
