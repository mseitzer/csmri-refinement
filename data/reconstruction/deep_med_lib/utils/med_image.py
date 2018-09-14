import numpy as np
from sklearn.feature_extraction import image
from sklearn.utils import check_random_state

__all__ = ['PatchExtractor3D',
           'extract_patches_3d',
           'compute_n_patches_nd']


# 3D Patch Extractor methods ####################
def compute_n_patches_nd(image_shape, patch_shape, max_patches=None):
    ''' Compute the number of patches that will be extracted in an image.

    Parameters
    ----------
    image_shape: tuple length m
        full image shape. i.e. for a list of 3D complex images,
        this is (10, 30, 2, 256, 256)
    patch_shape: tuple length n
        If the patch is 3d gray, it could be like (3, 6, 6).
        If it has channels, becomes (3, 2, 6, 6) for example
    max_patches: integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    '''

    # make sure two tuples are of same length
    pdim = len(patch_shape)
    n_samples = np.prod(image_shape[:-pdim])
    extracted_region_shape = image_shape[-pdim:]
    all_patches = n_samples * np.prod(map(lambda x, y: x - y + 1,
                                          extracted_region_shape, patch_shape))

    if max_patches:
        if (issubclass(type(max_patches), int) and max_patches <= all_patches):
            return max_patches
        elif (issubclass(type(max_patches), float) and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def extract_patches_3d(im, patch_size, max_patches=None, random_state=None):
    """Reshape a 3D image into a collection of patches

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : array, shape = (image_height, image_width, image_depth) or
        (image_height, image_width, image_depth, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a complex image with separate channels
        would have `n_channels=2`.

    patch_size : tuple of ints (patch_height, patch_width, patch_depth)
        the dimensions of one patch. (DO NOT PUT N_CHANNELS HERE)

    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state : int or RandomState
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.

    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width) or
         (n_patches, patch_height, patch_width, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.
    """

    for ix, px, s in zip(im.shape[:3], patch_size,
                         ['height', 'width', 'depth']):
        if px > ix:
            raise ValueError(''.join(["%s of the patch should be" % s,
                                      "less than the %s of the image." % s]))

    i_h, i_w, i_d = im.shape[:3]
    p_h, p_w, p_d = patch_size
    im = im.reshape((i_h, i_w, i_d, -1))
    n_colors = im.shape[-1]

    extracted_patches = image.extract_patches(im,
                                              patch_shape=patch_size +
                                              (n_colors, ),
                                              extraction_step=1)

    n_patches = compute_n_patches_nd((i_h, i_w, i_d), (p_h, p_w, p_d),
                                     max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        k_s = rng.randint(i_d - p_d + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, k_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, p_d, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w, p_d))
    else:
        return patches


class PatchExtractor3D():
    """Extracts 3D patches from a collection of images

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch

    max_patches : integer or float, optional default is None
        The maximum number of patches per image to extract. If max_patches is a
        float in (0, 1), it is taken to mean a proportion of the total number
        of patches.

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    """

    def __init__(self, patch_size=None, max_patches=None, random_state=None):
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.random_state = random_state

    def transform(self, X):
        """Transforms the image samples in X into a matrix of patch data.

        Parameters
        ----------
        X : array, shape = (n_samples, image_height, image_width) or
            (n_samples, image_height, image_width, n_channels)
            Array of images from which to extract patches. For color images,
            the last dimension specifies the channel: a RGB image would have
            `n_channels=3`.

        Returns
        -------
        patches: array, shape = (n_patches, patch_height, patch_width) or
             (n_patches, patch_height, patch_width, n_channels)
             The collection of patches extracted from the images, where
             `n_patches` is either `n_samples * max_patches` or the total
             number of patches that can be extracted.

        """
        self.random_state = check_random_state(self.random_state)
        n_images, i_h, i_w, i_d = X.shape[:4]
        X = np.reshape(X, (n_images, i_h, i_w, i_d, -1))
        n_channels = X.shape[-1]
        if self.patch_size is None:
            patch_size = i_h // 10, i_w // 10, i_d // 10
        else:
            patch_size = self.patch_size

        # compute the dimensions of the patches array
        p_h, p_w, p_d = patch_size
        n_patches = compute_n_patches_nd((i_h, i_w, i_d), patch_size,
                                         self.max_patches)
        patches_shape = (n_images * n_patches, ) + patch_size
        if n_channels > 1:
            patches_shape += (n_channels, )

        # extract the patches
        patches = np.empty(patches_shape)
        for ii, im in enumerate(X):
            patches[ii * n_patches:(ii + 1) * n_patches] = extract_patches_3d(
                im, patch_size, self.max_patches, self.random_state)
        return patches


def __interpolate_patch_shape(data_shape, desired_patch_shape):
    """
    Given desired patch shape to extract (i.e. patch considered at last axes),
    fills the preceeding dimensions with ones, based on the data shape.
    Used for patch extractor

    Example
    -------
    data_shape (1, 2, 30, 256, 256),
    desired_patch_shape (16, 16)
        --> returns (1, 1, 1, 16, 16)

    """
    n = len(data_shape)
    np = len(desired_patch_shape)
    new_patch_shape = (1, ) * (n - np) + desired_patch_shape
    return new_patch_shape
