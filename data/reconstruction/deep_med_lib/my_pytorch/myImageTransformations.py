import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.stats import truncnorm
import collections
import logging
from PIL import Image
import numbers
import cv2
from ..utils import compressed_sensing as cs
from ..utils import mymath
from ..utils import dnn_io

__author__ = "Wei OUYANG"
__license__ = "GPL"
__version__ = "0.1.0"
__status__ = "Development"


def get_mask_generator(sampling_scheme, im_shape, acceleration_factor,
                       variable=False, var_type='uniform', rng=None):
    if rng is None:
        rng = np.random

    logging.debug("sampling scheme: {}".format(sampling_scheme))

    size = im_shape[-1]

    def mask_gen():
        if sampling_scheme == 'radial':
            if variable:
                x_in = np.arange(1, size//2)

                # AGGRESSIVE version
                # For variable acceleration, we create the pdf as follows:
                # probability for each sampling rate is at most 0.5
                # probability for each sampling rate is at least 1/size_nx
                # in between we add exponential distribution which prefers lower acceleration factor
                # probability for each sampling rate is in between [1/size_nx, 1/2].
                if var_type == 'aggressive':
                    pdf = np.minimum(0.5, np.exp(-2*np.linspace(0, 4, len(x_in))) + 1./size)
                    pdf = pdf / np.sum(pdf)
                    acc_factors = rng.choice(x_in, im_shape[0], p=pdf)
                else:
                    # Uniform
                    acc_factors = rng.randint(1, len(x_in), im_shape[0])

                # Sample n times so acc_factor will be diff within the batch
                mask = []
                for i in range(im_shape[0]):
                    mask.append(cs.radial_sampling((1, size, size),
                                                   acc_factors[i],
                                                   rand=True,
                                                   golden_angle=True,
                                                   centred=False,
                                                   rng=rng))
                mask = np.array(mask)
                mask = mask.reshape(im_shape)

            else:
                # More efficient this way
                n_lines = acceleration_factor
                mask = cs.radial_sampling(im_shape,
                                          n_lines,
                                          rand=True,
                                          golden_angle=True,
                                          centred=False,
                                          rng=rng)
        else:
            central_lines = 8
            if variable:
                mask = np.zeros(im_shape)
                for i in range(im_shape[0]):
                    acc_r = float(rng.uniform(1, acceleration_factor*1.5))

                    mask[i] = cs.cartesian_mask(mask.shape[1:], acc_r, central_lines, centred=False, rng=rng)
            else:
                mask = cs.cartesian_mask(im_shape, acceleration_factor,
                                         central_lines, centred=False, rng=rng)
        return mask

    return mask_gen


def undersample(im, mask, rng=None):
    im_und, k_und = cs.undersample(im, mask, centred=False,
                                   norm='ortho', rng=rng)
    und_sampl_rate = np.sum(mask.ravel()) * 1. / mask.size
    return im_und, k_und, mask, und_sampl_rate


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def center_crop(x, center_crop_size):
    assert x.ndim == 3
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[:, centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]

def crop_image_at(image, cx, cy, sx, sy):
    # Crop a 3D image using a bounding box centred at (cx, cy)
    # and with specified size
    X, Y = image.shape[:2]
    r1, r2 = sx // 2, sy //2
    x1, x2 = cx - r1, cx + r1
    y1, y2 = cy - r2, cy + r2
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    crop = image[x1_: x2_, y1_: y2_]
    crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_)) + ((0, 0),) * (crop.ndim-2),
                  'constant')
    return crop



def to_tensor(x):
    import torch
    x = x.transpose((2, 0, 1))
    return torch.from_numpy(x).float()


def get_attribute(attribute, random_state):
    if isinstance(attribute, collections.Sequence):
        attr = random_num_generator(attribute, random_state=random_state)
    else:
        attr = attribute
    return attr


def random_num_generator(config, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        #print(config)
        raise Exception('unsupported format')
    return ret


def poisson_downsampling(image, peak, random_state=np.random):
    if not isinstance(image, np.ndarray):
        imgArr = np.array(image, dtype='float32')
    else:
        imgArr = image.astype('float32')
    Q = imgArr.max(axis=(0, 1)) / peak
    if Q[0] == 0:
        return imgArr
    ima_lambda = imgArr / Q
    noisy_img = random_state.poisson(lam=ima_lambda)
    return noisy_img.astype('float32')


def apply_gaussian_noise(im_in, mean=0, sigma=0.01):
    low_clip = -1. if im_in.min() < 0 else 0
    noise = np.random.normal(mean, sigma, im_in.shape)
    return np.clip(im_in + noise, low_clip, 255.)


def apply_poission_matlab(im_in):
    low_clip = -1. if im_in.min() < 0 else 0
    vals = len(np.unique(im_in))
    vals = 2 ** np.ceil(np.log2(vals))

    # Ensure image is exclusively positive
    if low_clip == -1.:
        old_max = im_in.max()
        im_in = (im_in + 1.) / (old_max + 1.)

    # Generating noise for each unique value in image.
    out = np.random.poisson(im_in * vals) / float(vals)

    # Return image to original range if input was signed
    if low_clip == -1.:
        out = out * (old_max + 1.) - 1.

    return np.clip(out, low_clip, 255.)


def apply_salt_and_pepper_noise(im_in, amount=0.1, salt_vs_pepper=0.5):
    out = im_in.copy()
    low_clip = -1. if im_in.min() < 0 else 0
    p = amount
    q = salt_vs_pepper
    flipped = np.random.choice([True, False], size=im_in.shape, p=[p, 1 - p])
    salted = np.random.choice([True, False], size=im_in.shape, p=[q, 1 - q])
    peppered = ~salted
    out[flipped & salted] = 255.
    out[flipped & peppered] = low_clip
    return np.clip(out, low_clip, 255.)


def apply_speckle_noise(im_in, mean=0, sigma=0.01):
    low_clip = -1. if im_in.min() < 0 else 0
    noise = np.random.normal(mean, sigma, im_in.shape)
    return np.clip(im_in + im_in * noise, low_clip, 255.)


def affine_transform(image, alpha_affine, borderMode=cv2.BORDER_CONSTANT):
    imshape = image.shape
    shape_size = imshape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    warped = cv2.warpAffine(image.reshape(shape_size + (-1,)), M, shape_size[::-1],
                            flags=cv2.INTER_NEAREST, borderMode=borderMode)

    #print(imshape, warped.shape)

    warped = warped[..., np.newaxis].reshape(imshape)

    return warped

def perspective_transform(image, alpha_warp):
    shape_size = image.shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]-square_size, center_square[1]+square_size],
                       center_square - square_size,
                       [center_square[0]+square_size, center_square[1]-square_size]])
    pts2 = pts1 + np.random.uniform(-alpha_warp, alpha_warp, size=pts1.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)


def elastic_transform(image, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    return result


def get_motion_blur_kernel(l=None, th=None):

    # create line kernel
    kernel_motion_blur = np.zeros((l, l))
    kernel_motion_blur[int((l-1)/2), :] = np.ones(l)
    kernel_motion_blur = kernel_motion_blur / l

    # rotate it
    M = cv2.getRotationMatrix2D((l/2,l/2), th, 1)
    kernel_motion_blur_rotated = cv2.warpAffine(kernel_motion_blur, M, (l, l))

    return kernel_motion_blur_rotated


def get_fspecial(length, angle):
    """Motion kernel is adopted from MATLAB's fspecial('motion')

    Challenge slide says:
       blur length: max ~25 px
       angle range: 0-180 degree
    """

    length = max(1, length)
    half = (length - 1) / 2# rotate half length around center
    phi = np.mod(angle, 180) / 180 * np.pi

    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    xsign = np.int16(np.sign(cosphi))
    linewdt = 1

    # define mesh for the half matrix, eps takes care of the right size
    # for 0 & 90 rotation
    eps = 2.2204e-16
    sx = np.fix(half * cosphi + linewdt * xsign - length * eps)
    sy = np.fix(half * sinphi + linewdt - length * eps)
    [x, y] = np.meshgrid(np.arange(0,np.int16(sx.flatten()[0])+(xsign*1),xsign),
                         np.arange(0,np.int16(sy.flatten()[0])+1))

    # define shortest distance from a pixel to the rotated line
    dist2line = (y * cosphi - x * sinphi)# distance perpendicular to the line

    rad = np.sqrt(x**2 + y**2)
    # find points beyond the line's end-point but within the line width
    #lastpix = np.where((rad >= half) and (np.abs(dist2line) <= linewdt))
    lastpix = np.logical_and((rad>=half),(np.abs(dist2line) <= linewdt))
    #distance to the line's end-point parallel to the line
    x2lastpix = half - np.abs((x[lastpix] + dist2line[lastpix] * sinphi) / cosphi)

    dist2line[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix ** 2)
    dist2line = linewdt + eps - np.abs(dist2line)
    dist2line[dist2line < 0] = 0# zero out anything beyond line width
    dist2line
    # unfold half-matrix to the full size
    kernel = np.rot90(dist2line,2)
    n_kernel = np.zeros((kernel.shape[0]+dist2line.shape[0]-1,
                         kernel.shape[1]+dist2line.shape[1]-1))
    n_kernel[0:kernel.shape[0],0:kernel.shape[1]] = kernel
    n_kernel[kernel.shape[0]-1:,kernel.shape[1]-1:] = dist2line
    #kernel(end + (mslice[1:end]) - 1, end + (mslice[1:end]) - 1).lvalue = dist2line
    n_kernel = n_kernel/(np.sum(n_kernel) + eps*length*length)

    if cosphi > 0:
        n_kernel = np.flipud(n_kernel)

    return n_kernel


# def apply_linear_motion_blur(im_in, kernel):
#     # Applied kernel to each color channel
#     im_out = np.zeros(im_in.shape)
#     for i in [0,1,2]:
#         im_out[:,:,i] = ndimage.convolve(im_in[:,:,i], kernel, mode='mirror', cval=0.0)

#     return im_out


def apply_linear_motion_blur(im_in, kernel):
    return cv2.filter2D(im_in, -1, kernel)


def out_of_focus_motion_blur(im_in, filter_size, sigma):
    """Out-of-focus/gaussian motion blur applied to im (equivalent of MATLAB's imfilter)
    # filterSize drawn from normal dist for now
    """
    # Truncate and sigma simultatnioulsy defiens the size of filter
    truncate = (((filter_size - 1)/2)-0.5)/sigma

    im_out = np.zeros(im_in.shape)
    for i in [0,1,2]:
        im_out[:,:,i] = ndimage.filters.gaussian_filter(input=im_in[:,:,i], sigma=sigma, truncate=truncate)

    return im_out


from scipy.ndimage import zoom
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * np.float32(h)))
    zw = int(np.round(zoom_factor * np.float32(w)))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor, zoom_factor)  + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # identity output
    else:
        out = img.copy()

    return out


def flip_classes(label, label_flip_rate=0.05):
    """Expects NxCxHxW"""
    label_in = label.copy()
    orig_shape = label_in.shape
    n_flip = int(label_in.size * label_flip_rate)
    n_class = len(np.unique(label))
    label_in = label_in.ravel()
    label_in[np.random.choice(label_in.size, n_flip, False)] = np.random.randint(0, n_class, n_flip)
    label_in = label_in.reshape(orig_shape)
    return label_in


def convert_to_1hot(label, n_class=4):
    # print('1hot', label.shape)
    # print('1hot, label values:', np.unique(label))
    # Convert a label map (H x W x 1) into a one-hot representation (H x W x C)
    label_flat = label.flatten().astype(np.int)
    n_data = len(label_flat)
    label_1hot = np.zeros((n_data, n_class), dtype='int16')
    label_1hot[range(n_data), label_flat] = 1
    label_1hot = label_1hot.reshape(tuple(label.shape[:-1]) + (4,))
    return label_1hot


def convert_from_1hot(label, axis=-1):
    return label.argmax(axis=axis)


class ConvertTo1Hot(object):
    def __init__(self, n_class=4):
        self.n_class = n_class

    def __call__(self, image):
        # print(image.shape, np.unique(image))
        return convert_to_1hot(image, self.n_class)


class Merge(object):
    """Merge a group of images
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, images):
        #print(images[0].shape, images[1].shape)
        if isinstance(images, collections.Sequence) or isinstance(images, np.ndarray):
            assert all([isinstance(i, np.ndarray)
                        for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s == shapes[0] for s in shapes]
                       ), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")


class Split(object):
    """Split images into individual arrays
    """

    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            if isinstance(s, collections.Sequence):
                slices_.append(slice(*s))
            else:
                slices_.append(s)
        assert all([isinstance(s, slice) for s in slices_]
                   ), 'slices must be consist of slice instances'
        self.slices = slices_
        self.axis = kwargs.get('axis', -1)

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                sl = [slice(None)] * image.ndim
                sl[self.axis] = s
                ret.append(image[sl])
            return ret
        else:
            raise Exception("obj is not an numpy array")


class AffineTransform(object):
    """Apply random affine transformation on a numpy.ndarray (H x W x C)

    Parameter:
    ----------

    alpha: Range [0, 4] seems good for small images
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, image):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        return affine_transform(image, self.alpha)


class PerspectiveTransform(object):
    """Apply random perspective transformation on a numpy.ndarray (H x W x C)

    Parameter:
    ----------

    alpha: Range [0, 2] seems good for small images
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, image):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        return perspective_transform(image, self.alpha)


class CoordinateTransform(object):
    """apply random Affine or Perspective Transformations"""

    def __init__(self, alpha_affine, alpha_persp, p, random_state=np.random):

        self.alpha_affine = alpha_affine
        self.alpha_persp = alpha_persp
        self.p = p
        self.random_state = random_state

    def __call__(self, image):
        alpha_affine = get_attribute(self.alpha_affine, self.random_state)
        alpha_persp = get_attribute(self.alpha_persp,   self.random_state)

        if (np.random.random() <= self.p):
            image = affine_transform(image, alpha_affine)
        else:
            image = perspective_transform(image, alpha_persp)

        return image


class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        return elastic_transform(image, alpha=alpha, sigma=sigma)


class AffineTransformPair(object):
    """Apply random affine transformation on a numpy.ndarray (H x W x C x 2)

    Parameter:
    ----------

    alpha: Range [0, 4] seems good for small images
    """

    def __init__(self, alpha, nc):
        self.alpha = alpha
        self.nc = nc

    def __call__(self, image):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha

        res = affine_transform(image, self.alpha)

        #print(res.shape)

        res[..., self.nc] = np.round(res[..., self.nc])

        #print(res.shape)

        return res



class PoissonSubsampling(object):
    """Poisson subsampling on a numpy.ndarray (H x W x C)
    """

    def __init__(self, peak, random_state=np.random):
        self.peak = peak
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.peak, collections.Sequence):
            peak = random_num_generator(
                self.peak, random_state=self.random_state)
        else:
            peak = self.peak
        return poisson_downsampling(image, peak, random_state=self.random_state)


class AddGaussianNoise(object):
    """Add gaussian noise to a numpy.ndarray (H x W x C)
    """

    def __init__(self, mean, sigma, random_state=np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.mean, collections.Sequence):
            mean = random_num_generator(
                self.mean, random_state=self.random_state)
        else:
            mean = self.mean
        row, col, ch = image.shape
        gauss = self.random_state.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        image += gauss
        return image


class AddSpeckleNoise(object):
    """Add speckle noise to a numpy.ndarray (H x W x C)
    """

    def __init__(self, mean, sigma, random_state=np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.mean, collections.Sequence):
            mean = random_num_generator(
                self.mean, random_state=self.random_state)
        else:
            mean = self.mean
        row, col, ch = image.shape
        gauss = self.random_state.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        image += image * gauss
        return image


class GaussianBlurring(object):
    """Apply gaussian blur to a numpy.ndarray (H x W x C)

    Equivalent to Matlab gaussian blur, if filter_size is specified

    sigma: the s.d. of the gaussian
    filter_size: determines the gaussian kernel size

    """

    def __init__(self, sigma, filter_size=None, random_state=np.random):
        self.sigma = sigma
        self.random_state = random_state
        self.truncate = 4. if not filter_size else (((filter_size - 1)/2)-0.5)/sigma

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma

        # Select the sigma from a uniform dist (Gaussian-ception)
        sigma_sample = np.random.uniform(low=0.0, high=sigma)
        return gaussian_filter(image, sigma=(sigma_sample, sigma_sample, 0),
                               truncate=self.truncate)


class MotionBlurring(object):
    """Apply motion blur to a numpy.ndarray (H x W x C)
    """

    def __init__(self, length, angle, random_state=np.random):
        self.length = length
        self.angle = angle
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.length, collections.Sequence):
            length = random_num_generator(
                self.length, random_state=self.random_state)
        else:
            length = self.length
        if isinstance(self.length, collections.Sequence):
            angle = random_num_generator(
                self.angle, random_state=self.random_state)
        else:
            angle = self.angle

        # specify parameters
        sd = 3
        l = int(get_truncated_normal(mean=length, sd=sd, low=1, upp=length+2*sd).rvs())
        th = np.random.randint(0, angle)

        kernel = get_fspecial(l, th)
        return cv2.filter2D(image, -1, kernel)


class AddVariousNoise(object):
    """Add Gaussian, Poission, S&P, and Speckle noise with some probabilities"""

    def __init__(self, gauss_mean, gauss_sigma, salt_amount, salt_vs_pepper,
                 speckle_mean, speckle_sigma, p=None, random_state=np.random):

        self.gauss_mean     = gauss_mean
        self.gauss_sigma    = gauss_sigma
        self.salt_amount    = salt_amount
        self.salt_vs_pepper = salt_vs_pepper
        self.speckle_mean   = speckle_mean
        self.speckle_sigma  = speckle_sigma
        self.p = [0.5] * 4 if not p else p
        self.random_state = random_state

    def __call__(self, image):
        p = self.p
        gauss_mean     = get_attribute(self.gauss_mean,     self.random_state)
        gauss_sigma    = get_attribute(self.gauss_sigma,    self.random_state)
        salt_amount    = get_attribute(self.salt_amount,    self.random_state)
        salt_vs_pepper = get_attribute(self.salt_vs_pepper, self.random_state)
        speckle_mean   = get_attribute(self.speckle_mean,   self.random_state)
        speckle_sigma  = get_attribute(self.speckle_sigma,  self.random_state)

        if (np.random.random() <= p[0]):
            image = apply_gaussian_noise(image, gauss_mean, gauss_sigma)

        if (np.random.random() <= p[1]):
            image = apply_poission_matlab(image)

        if (np.random.random() <= p[2]):
            image = apply_salt_and_pepper_noise(image, salt_amount, salt_vs_pepper)

        if (np.random.random() <= p[3]):
            image = apply_speckle_noise(image, speckle_mean, speckle_sigma)

        return image


class AddGaussianPoissonNoise(object):
    """Add poisson noise with gaussian blurred image to a numpy.ndarray (H x W x C)
    """

    def __init__(self, sigma, peak, random_state=np.random):
        self.sigma = sigma
        self.peak = peak
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.peak, collections.Sequence):
            peak = random_num_generator(
                self.peak, random_state=self.random_state)
        else:
            peak = self.peak
        bg = gaussian_filter(image, sigma=(sigma, sigma, 0))
        bg = poisson_downsampling(
            bg, peak=peak, random_state=self.random_state)
        return image + bg


class MaxScaleNumpy(object):
    """scale with max and min of each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, range_min=0.0, range_max=1.0):
        self.scale = (range_min, range_max)

    def __call__(self, image):
        mn = image.min(axis=(0, 1))
        mx = image.max(axis=(0, 1))
        return self.scale[0] + (image - mn) * (self.scale[1] - self.scale[0]) / (mx - mn)


class MedianScaleNumpy(object):
    """Scale with median and mean of each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, range_min=0.0, range_max=1.0):
        self.scale = (range_min, range_max)

    def __call__(self, image):
        mn = image.min(axis=(0, 1))
        md = np.median(image, axis=(0, 1))
        return self.scale[0] + (image - mn) * (self.scale[1] - self.scale[0]) / (md - mn)


class NormalizeNumpy(object):
    """Normalize each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __call__(self, image):
        image -= image.mean(axis=(0, 1))
        s = image.std(axis=(0, 1))
        s[s == 0] = 1.0
        image /= s
        return image

class InverseNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        for i, m, s in zip(range(len(tensor)), self.mean, self.std):
            tensor[i] = tensor[i] * s + m
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class MutualExclude(object):
    """Remove elements from one channel
    """

    def __init__(self, exclude_channel, from_channel):
        self.from_channel = from_channel
        self.exclude_channel = exclude_channel

    def __call__(self, image):
        mask = image[:, :, self.exclude_channel] > 0
        image[:, :, self.from_channel][mask] = 0
        return image


class RandomCropNumpy(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, random_state=np.random):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.random_state = random_state

    def __call__(self, img):
        w, h = img.shape[:2]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = self.random_state.randint(0, w - tw)
        y1 = self.random_state.randint(0, h - th)

        return img[x1:x1 + tw, y1: y1 + th, :]


class CenterCropNumpy(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        cx, cy = img.shape[0] // 2, img.shape[1] // 2
        res =  crop_image_at(img, cx, cy, self.size[0], self.size[1])
        #print("center crop", res.shape)
        return res


class HeartCenterCropNumpy(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        label = img[..., -1]

        im_centre = label.shape[0] // 2, label.shape[1] // 2
        cx, cy = map(lambda x, y: int(np.round(np.mean(x))) if np.any(x) else y, np.where(label > 0)[:2], im_centre)

        return crop_image_at(img, cx, cy, self.size[0], self.size[1])


class CenterCropInKspace(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        # expects [nx, ny] for now
        nx, ny = img.shape[:2]
        im_k = mymath.fft2c(img, axes=(0,1))
        im_k_cropped = crop_image_at(im_k, nx//2, ny//2, self.size[0], self.size[1])
        img_cropped = mymath.ifft2c(im_k_cropped, axes=(0,1))
        #print(img.shape, nx//2, ny//2, self.size, img_cropped.shape)
        return abs(img_cropped)


class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, input):
        h, w = input.shape[:2]
        th, tw = self.translation
        if tw == 0 and th == 0:
            return input
        tw = 0 if tw == 0 else np.random.randint(-tw, tw)
        th = 0 if th == 0 else np.random.randint(-th, th)
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1, x2 = max(0, tw), min(w + tw, w)
        y1, y2 = max(0, th), min(h + th, h)

        target = np.zeros_like(input)
        target[:y2-y1, :x2-x1] = input[y1:y2, x1:x2]
        return target

class RandomRotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) randomly
    """

    def __init__(self, angle_range=(0.0, 360.0), axes=(0, 1), mode='reflect', order=2, random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, image):
        angle = self.random_state.uniform(self.angle_range[0], self.angle_range[1])

        if isinstance(image, np.ndarray):
            mi, ma = image.min(), image.max()
            image = scipy.ndimage.interpolation.rotate(
                image, angle, reshape=False, axes=self.axes,
                mode=self.mode, order=self.order)
            return np.clip(image, mi, ma)
        elif isinstance(image, Image.Image):
            return image.rotate(angle)
        else:
            raise Exception('unsupported type')


class RandomRotatePair(object):
    """Rotate a pair of numpy.ndarray [(H x W x C), (H x W x k)]
    """

    def __init__(self, angle_range=(0.0, 360.0), axes=(0, 1), mode='reflect',
                 orders=[3, 0], nc=1, nk=1, random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode
        self.nc = nc
        self.nk = nk
        self.orders = orders


    def __call__(self, images):
        image = images[..., 0:self.nc]
        label = images[..., self.nc:]

        angle = self.random_state.uniform(self.angle_range[0], self.angle_range[1])

        if isinstance(image, np.ndarray):
            mi, ma = image.min(), image.max()
            image = scipy.ndimage.interpolation.rotate(image, angle, reshape=False,
                                                       axes=self.axes, mode=self.mode,
                                                       order=self.orders[0])
            image = np.clip(image, mi, ma)

            mi, ma = label.min(), label.max()
            label = scipy.ndimage.interpolation.rotate(label, angle, reshape=False,
                                                       axes=self.axes, mode=self.mode,
                                                       order=self.orders[1])
            label = np.clip(label, mi, ma)

            return np.concatenate([image, label], axis=-1)

        elif isinstance(image, Image.Image):
            return image.rotate(angle)
        else:
            raise Exception('unsupported type')


class RandomZoom(object):
    '''pars:
            zoom: (float, float) lower and upper bounds for the zoom factor
    '''
    def __init__(self, zoom, order=2):
        self.zoom = zoom
        self.order = order

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            sampled_zoom = np.random.uniform(low=self.zoom[0], high=self.zoom[1])
            return clipped_zoom(image, sampled_zoom, order=self.order)
        else:
            raise Exception('unsupported type')


class RandomZoomPair(object):
    '''pars:
            zoom: (float, float) lower and upper bounds for the zoom factor
    '''
    def __init__(self, zoom, orders, nc):
        self.zoom = zoom
        self.orders = orders
        self.nc = nc

    def __call__(self, images):
        if isinstance(images, np.ndarray):
            sampled_zoom = np.random.uniform(low=self.zoom[0], high=self.zoom[1])
            image = images[..., :self.nc]
            label = images[..., self.nc:]
            image = clipped_zoom(image, sampled_zoom, order=self.orders[0])
            label = clipped_zoom(label, sampled_zoom, order=self.orders[1])

            return np.concatenate([image, label], axis=-1)
        else:
            raise Exception('unsupported type')


class BilinearResize(object):
    """Resize a PIL.Image or numpy.ndarray (H x W x C)
    """

    def __init__(self, zoom):
        self.zoom = [zoom, zoom, 1]

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return scipy.ndimage.interpolation.zoom(image, self.zoom)
        elif isinstance(image, Image.Image):
            return image.resize(self.size, Image.BILINEAR)
        else:
            raise Exception('unsupported type')


class FlipClassLabels(object):
    """Flip class labels with probability p"""

    def __init__(self, p, random_state=np.random):
        self.p = p
        self.random_state = random_state

    def __call__(self, image):
        p = np.random.uniform(low=0.0, high=self.p*2)
        image = flip_classes(image, p)
        return image


class UndersampleWithResizedGrid(object):
    def __init__(self, mask_type, acceleration_rate=4, variable=False,
                 grid_resize=210, n=0):
        self.size = grid_resize
        self.mask_shape = (max(1,n), self.size, self.size)
        self.mask_gen = get_mask_generator(mask_type, self.mask_shape,
                                           acceleration_rate, variable)

    def __call__(self, image):
        # Image should have size Nx x Ny [x Nt]
        #print ('undersample', image.shape)
        nx, ny = image.shape[:2]

        if nx == self.size and ny == self.size:  # no resizing required
            mask = self.mask_gen()
            image = np.squeeze(image)
            orig_shape = image.shape
            image = image.reshape(self.size, self.size, -1).transpose((2,0,1))
            im_und, k_und, mask, _ = undersample(image, mask)
            im_und = im_und.transpose((1,2,0))
        else:

            image = np.squeeze(image)
            # crop_resize back
            image = crop_image_at(image, nx//2, ny//2, self.size, self.size)
            mask = self.mask_gen()

            # reshape
            orig_shape = image.shape
            image = image.reshape(self.size, self.size, -1).transpose((2,0,1))
            im_und, k_und, mask, _ = undersample(image, mask)

            im_und = im_und.transpose((1,2,0)).reshape(orig_shape)


            # crop_resize back
            im_und = crop_image_at(im_und, self.size// 2, self.size// 2, nx, ny)

        return np.stack([np.real(im_und), np.imag(im_und)], axis=len(im_und.shape))


# class UndersampleWithResizedGrid(object):
#     def __init__(self, mask_type, acceleration_rate=4, variable=False,
#                  grid_resize=210, n=0):
#         self.size = grid_resize
#         self.mask_shape = (max(1,n), self.size, self.size)
#         self.mask_gen = get_mask_generator(mask_type, self.mask_shape,
#                                            acceleration_rate, variable)
#         print(self.mask_shape)

#     def __call__(self, image):
#         # Image should have size Nx x Ny [x Nt]
#         #print ('undersample', image.shape)
#         image = np.squeeze(image)

#         if image.ndim == 3:
#             image = image.transpose((2, 0, 1))
#             mask = self.mask_gen()
#             im_und, _, _, _ = undersample(image, mask)
#             im_und = im_und.transpose((1, 2, 0))
#         else:
#             mask = self.mask_gen()
#             print(mask.shape)
#             # crop
#             nx, ny = image.shape[:2]
#             image = crop_image_at(image, nx//2, ny//2, self.size, self.size)
#             orig_shape = image.shape
#             image = image.reshape(self.size, self.size, -1).transpose((2,0,1))

#             print(image.shape)

#             im_und, k_und, mask, _ = undersample(image, mask)

#             # crop back
#             im_und = im_und.transpose((1,2,0)).reshape(orig_shape)
#             im_und = crop_image_at(im_und, self.size // 2, self.size // 2, nx, ny)

#         return np.array([np.real(im_und), np.imag(im_und)]).transpose((1,2,0))


class Undersample(object):
    def __init__(self, mask_type, im_shape, acceleration_rate=4, variable=False,
                 fixed_mask=False, num_fixed_masks=1):
        if fixed_mask:
            self.rng = np.random.RandomState(seed=0)
            mask_gen = get_mask_generator(mask_type, im_shape,
                                          acceleration_rate,
                                          variable, rng=self.rng)
            # This does not work as expected when using multiple workers
            # to preprocess data! Use only one thread during validation.
            self.current_mask = 0
            self.fixed_masks = [mask_gen() for _ in range(num_fixed_masks)]
        else:
            self.rng = np.random
            self.mask_gen = get_mask_generator(mask_type, im_shape,
                                               acceleration_rate,
                                               variable, rng=self.rng)
            self.fixed_masks = None

    def __call__(self, image):
        # Image should have size Nx x Ny [x Nt]
        #print ('undersample', image.shape)
        image = image.transpose((2, 0, 1))

        if self.fixed_masks is None:
            mask = self.mask_gen()
        else:
            mask = self.fixed_masks[self.current_mask].copy()
            self.current_mask = (self.current_mask + 1) % len(self.fixed_masks)

        #print(image.shape, mask.shape)
        im_und, k_und, mask, _ = undersample(image, mask, self.rng)

        grp = np.concatenate([dnn_io.to_tensor_format(im_und),
                              dnn_io.to_tensor_format(k_und),
                              dnn_io.to_tensor_format(mask, mask=True),
                              # TODO if we wish to crop, need to think about fftshifting
                              # dnn_io.to_tensor_format(np.fft.fftshift(k_und, axes=(-1,-2))),
                              # dnn_io.to_tensor_format(np.fft.fftshift(mask, axes=(-1,-2)), mask=True),
                              dnn_io.to_tensor_format(image)], axis=1)
        grp = grp.squeeze().transpose((1, 2, 0))

        return grp


class EnhancedCompose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(
                    t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img


if __name__ == '__main__':
    from torchvision.transforms import Lambda

    input_channel = 3
    target_channel = 3

    # define a transform pipeline
    transform = EnhancedCompose([
        Merge(),
        RandomCropNumpy(size=(512, 512)),
        RandomRotate(),
        Split([0, input_channel], [input_channel, input_channel + target_channel]),
        [CenterCropNumpy(size=(256, 256)), CenterCropNumpy(size=(256, 256))],
        [NormalizeNumpy(), MaxScaleNumpy(0, 1.0)],
        # for non-pytorch usage, remove to_tensor conversion
        [Lambda(to_tensor), Lambda(to_tensor)]
    ])
    # read input data for test
    #image_name = '/home/ozanoktay/data/deblurchallenge_data/train/0ac23e11-1f5d-4f75-8328-7b6d2be84a61-0.jpg'
    image_name = '/home/js3611/projects/deblurchallenge/data_dir/train/0ac23e11-1f5d-4f75-8328-7b6d2be84a61-0.jpg'
    image_in = np.array(Image.open(image_name))
    image_target = np.array(Image.open(image_name))

    # apply the transform
    x, y = transform([image_in, image_target])
