__author__ = 'js3611'

import numpy as np
import scipy.stats as ss
from scipy.ndimage.filters import gaussian_laplace
from skimage.measure import compare_ssim 

def mse(x, y):
    return np.mean(np.abs(x - y)**2)


def nmse(x, y):
    return np.sum(np.abs(x - y)**2) / np.sum(np.abs(x)**2)


def psnr(x, y):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of either integer (0, 256) or float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float) \
        and np.issubdtype(y.dtype, np.float)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def complex_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max

    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).

    '''
    mse = np.mean(np.abs(x - y)**2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse)


def HFEN(x, y, sigma=1.5, truncate=4.5):
    """High Frequency Error Norm

    Works for any dimension, but needs to be float. Gaussian kernel size is defined by:

    2*int(truncate*sigma + 0.5) + 1. so if

    15x15 Gauss kernel with sigma 1.5: truncate=4.5, sigma=1.5
    13x13 Gauss kernel with sigma 1.5: truncate=4, sigma=1.5

    Parameters:
    ----------

    x: reference
    y: reconstruction

    Returns:
    --------

    hfen: scalar
    """
    x_log = gaussian_laplace(abs(x), sigma)
    y_log = gaussian_laplace(abs(y), sigma)

    return np.linalg.norm(x_log.reshape(-1) - y_log.reshape(-1)) / np.linalg.norm(x_log)


def ssim(x, y):
    """
    Note that ssim only takes 2D image, so the data needs to be of form (n1, n2, ..., nx, ny)
    """
    nx, ny = x.shape[-2:]
    xs = x.reshape(-1, nx, ny)
    ys = y.reshape(-1, nx, ny)

    ssim_res = []
    for xi, yi in zip(xs, ys):
        res = compare_ssim(np.abs(xi), np.abs(yi))
        #data_range=(np.abs(x).max() - np.abs(x).min()))
        ssim_res.append(res)

    return np.mean(ssim_res)


def categorical_dice(prediction, truth, k):
    # Dice overlap metric for label value k
    A = prediction.astype(int) == k
    B = truth.astype(int) == k
    return 2. * np.sum(A * B) / (np.sum(A) + np.sum(B))
