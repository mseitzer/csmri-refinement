import numpy as np
from . import mymath
#import mymath
from numpy.lib.stride_tricks import as_strided


def soft_thresh(u, lmda):
    Su = (abs(u) - lmda) / abs(u) * u
    Su[abs(u) < lmda] = 0
    return Su


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def var_dens_mask_2d_unif(shape,
                       ivar_max,
                       sample_high_freq=True,
                       baseline_sensitivity=None):
    '''
    need 3D
    :param shape:
    :param ivar:
    :return:
    '''
    mask = np.zeros(shape)
    Nt, Nx, Ny = shape

    ivars = np.random.uniform(size=Nt) * ivar_max
    if not baseline_sensitivity:
        baseline_sensitivity = get_undersampling_sensitivity(
            (Nx, Ny), 100. / 95)

    for t in range(Nt):
        ivar = ivars[t]
        pdf_x = normal_pdf(Nx, ivar)
        pdf_y = normal_pdf(Ny, ivar)
        pdf = np.outer(pdf_x, pdf_y)
        mask[t] = pdf
        if sample_high_freq and ivar > baseline_sensitivity:
            mask[t] = mask[t] / 1.1 + 0.01

    mask = np.random.binomial(1, mask)
    xc = Nx / 2
    yc = Ny / 2
    mask[:, xc - 5:xc + 5, yc - 5:yc + 5] = 1

    return mask


def var_dens_mask_2d(shape, ivar, sample_high_freq=True):
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    pdf_x = normal_pdf(Nx, ivar)
    pdf_y = normal_pdf(Ny, ivar)
    pdf = np.outer(pdf_x, pdf_y)

    size = pdf.itemsize
    strided_pdf = as_strided(pdf, (Nt, Nx, Ny), (0, Ny * size, size))
    # this must be false if undersampling rate is very low (around 90%~ish)
    if sample_high_freq:
        strided_pdf = strided_pdf / 1.1 + 0.01
    mask = np.random.binomial(1, strided_pdf)

    xc = Nx / 2
    yc = Ny / 2
    mask[:, xc - 4:xc + 5, yc - 4:yc + 5] = True

    if Nt == 1:
        return mask.reshape((Nx, Ny))

    # what is this?
    # acc = Nx*Ny*Nt / sum(sum(sum(mask_batch(:,:,:))));
    return mask


def cartesian_mask(shape, acc, sample_n=10, centred=False, rng=None):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be 4,8, 10, etc..

    """
    if rng is None:
        rng = np.random

    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = Nx // acc

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    # strided_pdf = as_strided(pdf_x, (Nt, Nx, 1), (0, size, 0))
    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = rng.choice(Nx, int(n_lines), False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask


def cartesian_mask2(shape, ivar, centred=False,
                   sample_high_freq=True, sample_centre=True, sample_n=10):
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    pdf_x = normal_pdf(Nx, ivar)

    # this must be false if undersampling rate is very low (around 90%~ish)
    if sample_high_freq:
        # pdf_x = pdf_x / 1.1 + 0.01
        pdf_x = pdf_x / 1.25 + 0.02

    size = pdf_x.itemsize
    strided_pdf = as_strided(pdf_x, (Nt, Nx, 1), (0, size, 0))
    mask = np.random.binomial(1, strided_pdf)
    size = mask.itemsize
    mask = as_strided(mask, (Nt, Nx, Ny), (size * Nx, size, 0))

    if sample_centre:
        s = sample_n / 2
        xc = Nx / 2
        yc = Ny / 2
        mask[:, xc - s:xc - s + sample_n, :] = 1

    if Nt == 1:
        return mask.reshape((Nx, Ny))

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask


def cartesian_mask_guarantee_sampling_rate(shape,
                                           acc_rate,
                                           ivar,
                                           tol=0.1,
                                           centred=False,
                                           sample_high_freq=True,
                                           sample_centre=True,
                                           sample_n=10,):
    """Guarantee acc_rate +/- 10%. Temporary function"""

    max_attempt = 10
    mask = np.zeros(shape)
    idx = 0
    ctr = 0
    n_mask = len(mask)
    while idx < n_mask and ctr < max_attempt:
        im_shape = (n_mask*100, ) + shape[1:]
        curr_mask = cartesian_mask(im_shape,
                                   ivar,
                                   centred=centred,
                                   sample_high_freq=sample_high_freq,
                                   sample_centre=sample_centre,
                                   sample_n=sample_n)
        for m in curr_mask:
            curr_rate = m.size / float(np.sum(m))
            if acc_rate * (1-tol) < curr_rate < acc_rate * (1+tol):
                #print('y' + ' {}'.format(curr_rate))
                mask[idx] = m
                idx += 1
            # else:
                #print('n' + ' {}'.format(curr_rate))
            if idx >= n_mask:
                break

        ctr += 1

    if idx < n_mask:
        mask[idx:n_mask] = curr_mask[idx:n_mask]

    return mask


def nlines(shape,
           n,
           centred=False,
           sample_centre=True,
           sample_n=1):
    """ Sample n lines uniformly"""
    Nt, Nx, Ny = shape
    mask = np.zeros(shape)
    # print mask.shape
    if sample_centre:
        xc = Nx / 2
        yc = Ny / 2
        s = sample_n / 2
        mask[:, xc - s:xc - s + sample_n, :] = 1
        n -= sample_n

    if n == 0:
        return mask

    for t in range(Nt):
        idx = np.random.choice(Nx, n, replace=False)
        if sample_centre:
            while len(set(idx.tolist()).intersection(set(np.r_[xc - s:xc - s + sample_n]))) != 0:
                idx = np.random.choice(Nx, n, replace=False)
        mask[t, idx, :] = 1

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask


def lowres(shape,
           n,
           centred=False,
           partial=True,
           skip=True):
    """ Sample n lines uniformly"""
    Nt, Nx, Ny = shape
    mask = np.zeros((Nt, Nx, 1))
    # print mask.shape
    xc = Nx / 2
    yc = Ny / 2
    for t in range(Nt):
        if skip:
            s = n
            sign = ((np.random.binomial(1, 0.5, s) - 0.5)*2).astype('int')
            mask[t, xc + np.arange(n) * sign, :] = 1
        else:
            s = n/2
            mask[t, xc-s: xc-s+n, :] = 1

    mask = np.repeat(mask,Ny, axis=-1)

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask




def one_line(shape):
    #print(shape)
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    mask = np.zeros_like(shape).astype(bool)
    xc = Nx / 2

    print(mask.shape)
    print(xc)

    mask[:, xc, :] = True

    if Nt == 1:
        return mask.reshape((Nx, Ny))

    return mask


def shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                    centred=False, sample_n=10):
    '''
    Creates undersampling mask which samples in sheer grid

    Parameters
    ----------

    shape: (nt, nx, ny)

    acceleration_rate: int

    Returns
    -------

    array

    '''
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in range(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    xc = Nx / 2
    xl = sample_n / 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        if xl > 0:
            mask[:, :xl] = 1
        if xh > 0:
            mask[:, -xh:] = 1

    mask_rep = np.repeat(mask[..., np.newaxis], Ny, axis=-1)
    return mask_rep


def perturbed_shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                              centred=False,
                              sample_n=10):
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in range(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    # brute force
    rand_code = np.random.randint(0, 3, size=Nt*Nx)
    shift = np.array([-1, 0, 1])[rand_code]
    new_mask = np.zeros_like(mask)
    for t in range(Nt):
        for x in range(Nx):
            if mask[t, x]:
                #print('(%d, %d) * [%d] -> (%d, %d)' % (t, x, shift[t*x], t, (x + shift[t*x])%Nx))
                new_mask[t, (x + shift[t*x])%Nx] = 1

    xc = Nx / 2
    xl = sample_n / 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        new_mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        new_mask[:, :xl] = 1
        new_mask[:, -xh:] = 1

    mask_rep = np.repeat(new_mask[..., np.newaxis], Ny, axis=-1)
    # print(start)
    # print(shift.reshape(Nt, Nx))



    return mask_rep


def get_undersampling_ratio(dim,
                            undersampling_sensitivity,
                            Nz=20,
                            gen_mask=var_dens_mask_2d, **kwargs):
    '''

    Parameters
    ----------
    dim: 2-tuple chape

    undersampling_sensitivity: the inverse of standard deviation (sensitivity)


    Returns
    -------
        mean_rate: mean ratio of (undersampled data) : (fully sampled data)
        std_rate: std deviation of the above
    '''
    Nx, Ny = dim
    mask = gen_mask([Nz, Nx, Ny], undersampling_sensitivity, **kwargs)
    undersampling_ratio = np.zeros([Nz, 1])

    # get subsampling rate per mask in z direction, get standard deviation
    for i in range(Nz):
        undersampling_ratio[i] = np.sum(mask[i],
                                        dtype=float) / mask[i, :, :].size

    mean_rate = np.mean(undersampling_ratio)
    std_rate = np.std(undersampling_ratio)
    return mean_rate, std_rate


def get_undersampling_sensitivity(dim,
                                  undersampling_factor=4,
                                  gen_mask=var_dens_mask_2d,
                                  **kwargs):
    '''
    Gets gauss_ivar for subsampling to mean+/- std*1e-2 range

    dim: [Nx, Ny]
    undersampling_factor: default: 4 (i.e. 4 times undersampling)

    returns:
        sensitivity_rate

    '''

    target_rate = 1. / undersampling_factor
    sensitivity = 1e-10
    lb = sensitivity
    stepsize = 1e-10
    tol = 1e-5
    for factor in np.arange(10, 0, -1):
        print('factor {}'.format(factor))
        ub = sensitivity
        while True:
            mean_r, std_r = get_undersampling_ratio(
                dim, ub, Nz=100,
                gen_mask=gen_mask, **kwargs)
            if mean_r - tol < target_rate:
                sensitivity = lb
                break
            if ub > 2:
                msg = (
                    'We cant find the sensitivity for given undsampl. factor.'
                    'i.e. val is too low.'
                    ' Some masks sample central n-lines. Ensure that'
                    '(central samples)/mask.size > 1/undersampling_factor ')
                raise Exception(msg)

            print('target: {}, mean+/-std: {} +/- P{}'.format(target_rate, mean_r, std_r))

            # at this point, mean - std using upper bound is lower than the target rate
            lb = ub
            ub += stepsize * np.power(10, factor)
            print('Lower Bound: {}, Upper Bound: {}'.format(lb, ub))

        if np.abs(target_rate - mean_r) <= std_r * tol:
            break
    return sensitivity


def undersample(x, mask, centred=False, norm='ortho', noise=0, rng=None):
    '''
    Undersample x. FFT2 will be applied to the last 2 axis
    Parameters
    ----------
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain

    norm: 'ortho' or None
        if 'ortho', performs unitary transform, otherwise normal dft

    noise_power: float
        simulates acquisition noise, complex AWG noise.
        must be percentage of the peak signal

    Returns
    -------
    xu: array_like
        undersampled image in image domain. Note that it is complex valued

    x_fu: array_like
        undersampled data in k-space

    '''
    if rng is None:
        rng = np.random

    assert x.shape == mask.shape, \
        'Shape of x {} does not match mask shape {}'.format(x.shape,
                                                            mask.shape)
    # zero mean complex Gaussian noise
    noise_power = noise
    nz = np.sqrt(.5)*(rng.normal(0, 1, x.shape) + 1j * rng.normal(0, 1, x.shape))
    nz = nz * np.sqrt(noise_power)

    if norm == 'ortho':
        # multiplicative factor
        nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
    else:
        nz = nz * np.prod(mask.shape[-2:])

    if centred:
        x_f = mymath.fft2c(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = mymath.ifft2c(x_fu, norm=norm)
        return x_u, x_fu
    else:
        x_f = mymath.fft2(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = mymath.ifft2(x_fu, norm=norm)
        return x_u, x_fu


def data_consistency(x, y, mask, centered=False, norm='ortho'):
    '''
    x is in image space,
    y is in k-space
    '''
    if centered:
        xf = mymath.fft2c(x, norm=norm)
        xm = (1 - mask) * xf + y
        xd = mymath.ifft2c(xm, norm=norm)
    else:
        xf = mymath.fft2(x, norm=norm)
        xm = (1 - mask) * xf + y
        xd = mymath.ifft2(xm, norm=norm)

    return xd


def data_consistency_xf(x, xk, mask, centered=False, norm='ortho'):
    '''
    x is in x-f space, shape assumed to be [n, nt, nx, ny]
    xk is in k-t space

    assume it is always centred transform along t/f

    '''
    xt = mymath.fftc(x, axis=1, norm=norm)
    kt = mymath.fft2(xt, norm=norm)
    ktm = (1 - mask) * kt + xk
    xt_post = mymath.ifft2(ktm, norm=norm)
    xf_post = mymath.ifftc(xt_post, axis=1, norm=norm)

    return xf_post


def get_phase(x):
    xr = np.real(x)
    xi = np.imag(x)
    phase = np.arctan(xi / (xr + 1e-12))
    return phase


def genD(Nt, Nx, Ny):
    pass


def denoise_tv(y, lmda, n_iter, D, Dt):
    pass


def undersampling_rate(mask):
    return float(mask.sum()) / mask.size


def radial_sampling(shape, n_lines,
                    angle_begin=0, rand=False, golden_angle=False, centred=True, rng=None):
    """ Radial sampling, gridded to the nearest cartesian coordinate.
    For now only supports when nx=ny

    ===========================================================================
    Note: Code inspired by strucrand from k-t SLR:

    Lingala SG1, Hu Y, DiBella E, Jacob M.,
    "Accelerated dynamic MRI exploiting sparsity and low-rank structure: k-t SLR",
    TMI May, 2011
    ===========================================================================

    Parameters
    ----------

    shape: (..., Nt, Nx, Ny)

    n_lines: int

    angle_begin: where the sampling begins. ONLY for golden angle

    rand: bool - if True, angle_begin will be set randomly [0, pi).
                 ONLY for golden angle. for uniform, noise is always added

    golden_angle: bool - if True, use golden angle rule for each spoke,
                         otherwise uniform

    """
    if rng is None:
        rng = np.random

    GOLDEN_ANGLE = np.pi / ((1 + np.sqrt(5)) / 2)

    n, nx0, ny0 = np.prod(shape[:-2]), shape[-2], shape[-1]
    nx = ny = max(nx0, ny0)

    assert nx == ny

    mask = np.zeros((n, nx, ny), dtype=int)

    if rand:
        angle_begin = np.pi * rng.random()

    y = np.arange(-nx/2, nx/2, 1)
    x = np.arange(-ny/2, ny/2, 1)

    if golden_angle:
        angles = [angle_begin + i * GOLDEN_ANGLE for i in range(n_lines * n)]
    else:
        angles = np.tile(np.arange(0, np.pi, np.pi/n_lines), n)
        # Add offset increments
        angles += np.repeat(rng.random(n) * np.pi/n_lines, n_lines)

    kloc_all = np.outer(y, np.cos(angles)) + 1j * np.outer(x, np.sin(angles))

    # Round the collected data to the nearest cartesian location
    # & recentre at (n1/2, n2/2)
    kloc1 = np.round(kloc_all + (0.5 + 0.5*1j))+((nx/2)+(ny/2)*1j)
    kloc1real = np.real(kloc1)
    kloc1real = kloc1real - nx*(kloc1real > nx)
    kloc1imag = np.imag(kloc1)
    kloc1imag = kloc1imag - ny*(kloc1imag > ny)
    kloc1real = kloc1real + nx*(kloc1real < 1)
    kloc1imag = kloc1imag + ny*(kloc1imag < 1)
    kloc1 = kloc1real + 1j*kloc1imag
    t = np.repeat(np.arange(n), n_lines * nx)
    x = (np.real(kloc1.transpose().reshape(-1))-1).astype(int)
    y = (np.imag(kloc1.transpose().reshape(-1))-1).astype(int)
    mask[t, x, y] = 1

    if nx0 != ny0:
        # pad them
        xpad = (nx - nx0) / 2
        ypad = (ny - ny0) / 2
        mask = mask[:, xpad:nx-xpad, ypad:ny-ypad]

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-2, -1))
    return mask.reshape(shape)
