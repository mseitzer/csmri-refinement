import numpy as np
import numpy.linalg as linalg


def centering(X):
    '''
    Removes the mean intensity of the image from each image
    :param X: [m,n], image in m dimension, n samples
    :return: np.dot(np.identity(m) - np.ones([m,m]) / float(m) , X)
    '''
    # X: [m,n]
    m, n = X.shape
    # mathematically
    # return
    # prob faster:
    return X - np.mean(X, axis=0)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def contrast_normalization(X, c=0.2):
    # X: [m,n]
    return X / np.maximum(c, linalg.norm(X, axis=0))


def scale2unit(x):
    return x / np.max(x)


def whitening(X, eps=10e-6):
    # X = [m,n]
    # returns transformation for mapping
    X = X - np.array([np.mean(X, axis=1)]).T
    covX = np.dot(X,X.T) / len(X)
    
    w, v = linalg.eig(covX)
    w = np.real(w)
    idx = w > eps
    w = np.sqrt(w[idx])
    v = np.real(v[:,idx])

    ## sort by decreasing order of eigen vals
    #     idx = w.argsort()[::-1]
    #     w = w[idx]
    #     v = v[:,idx]
    w_inv = 1 / w 

    transform = np.dot(np.dot(v, np.diag(w_inv)), v.T)
    
    return transform


def extract_patches(img, patch_shape=(3, 3), stride=1):
    patches = []    
    for i in xrange(0,(img.shape[0]-patch_shape[0]+1), stride):
        for j in xrange(0,(img.shape[1]-patch_shape[1]+1), stride):
            patch = img[i:i+patch_shape[0], j:j+patch_shape[1]]
            patches.append(patch)
    
    return np.array(patches).reshape(-1, 1, patch_shape[0], patch_shape[1])


def assemble_patches(patches, img_shape, stride=1, pad=(0,0)):
    if len(patches) == 0:
        return
    p0, p1 = patches[0].shape
    k = 0
    n_patches = len(patches)
    img = np.zeros(img_shape)
    overlap_ctr = np.zeros(img_shape)
    for i in xrange(0, img_shape[0]-p0+1-pad[0], stride):
        for j in xrange(0, img_shape[1]-p1+1-pad[0], stride):
            if k >= n_patches:
                break
            img[i:i+p0, j:j+p1] += patches[k]
            overlap_ctr[i:i+p0, j:j+p1] += 1
            k += 1
    return img / overlap_ctr


def assemble_patches_rev(patches, img_shape, stride=1, pad=(0,0)):
    if len(patches) == 0:
        return
    p0, p1 = patches[0].shape
    k = 0
    n_patches = len(patches)
    img = np.zeros(img_shape)
    overlap_ctr = np.zeros(img_shape)
    for i in xrange(0, img_shape[0]-p0+1-pad[0], stride):
        for j in xrange(0, img_shape[1]-p1+1-pad[0], stride):
            if k >= n_patches:
                break
            img[j:j+p1,i:i+p0] += patches[k]
            overlap_ctr[j:j+p1,i:i+p0] += 1
            k += 1
    return img / overlap_ctr


def visualise_patches(patches, tile_shape=None, padding=True, pad_val=0):
    '''

    Parameters
    ----------
    patches: 3D tensor 

    tile_shape: iterable of 2 elements, [t0, t1]
         specify how to tile the patches to visualize. By default, it will be a square

    '''
    # get dimension of the patch
    # print patches.shape
    n = len(patches)
    p0, p1 = patches[0].shape
    if not tile_shape:
        # infer tile shape
        t0 = int(np.floor(np.sqrt(n))+1)
        t1 = t0
    t0, t1 = tile_shape
    #print patches
    # pad the array

    v_padding = t0 if padding else 0
    h_padding = t1 if padding else 0
    vismat = np.zeros((p0 * t0 + v_padding, p1 * t1 + h_padding))
    vismat[...] = pad_val
    # print vismat.shape
    for i in xrange(min(t0*t1, n)):
        # find place to insert
        cpad = (i % t1) if padding else 0
        rpad = (i / t1) if padding else 0
        ridx = (i / t1) * p0
        cidx = (i % t1) * p1
        # print i, cidx, ridx
        vismat[ridx+rpad:ridx+p0+rpad, cidx+cpad:cidx+p1+cpad] = patches[i]
    return vismat


def crop2d(images, crop_shape):
    '''
    Crop image from the center of the image.
    :param images: 4d numpy array, (#data, channel (depth), height, width)
    :param crop_shape: shape (h, w)
    :return: cropped_images
    '''
    img_shape = images.shape
    x_center = img_shape[2] / 2
    y_center = img_shape[3] / 2
    crop_h = crop_shape[0]
    crop_w = crop_shape[1]
    start_h = x_center - crop_h/2
    start_w = y_center - crop_w/2

    def crop(x):
        return x[0, start_h:start_h+crop_h, start_w:start_w+crop_w]

    cropped_images = [crop(img) for img in images]
    return np.array(cropped_images).reshape(-1, 1, crop_shape[0], crop_shape[1])


def crop3d(images, crop_shape):
    '''
    Crop image from the center of the image.
    :param images: 4d numpy array, (#data, depth, height, width)
    :param crop_shape: shape (d, h, w)
    :return: cropped_images
    '''
    img_shape = images.shape
    z_center = img_shape[1] / 2
    x_center = img_shape[2] / 2
    y_center = img_shape[3] / 2
    crop_d = crop_shape[0]
    crop_h = crop_shape[1]
    crop_w = crop_shape[2]
    start_d = z_center - crop_d/2
    start_h = x_center - crop_h/2
    start_w = y_center - crop_w/2

    def crop(x):
        return x[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]

    cropped_images = [crop(img) for img in images]
    return np.array(cropped_images)


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
