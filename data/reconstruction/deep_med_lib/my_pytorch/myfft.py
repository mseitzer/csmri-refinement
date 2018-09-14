import torch
import numpy as np

# Need pip package pytorch-fft. If this fails during install with "cuda.h no
# such file or directory", try to pass the path to the cuda include folder with
# the CPATH env variable
from pytorch_fft.fft import fft,ifft,fft2,ifft2,fft3,ifft3,rfft,irfft,rfft2,irfft2,rfft3,irfft3


def make_contiguous(*Xs):
    return tuple(X if X.is_contiguous() else X.contiguous() for X in Xs)


def contiguous_clone(X):
    if X.is_contiguous():
        return X.clone()
    else:
        return X.contiguous()


class Fft(torch.autograd.Function):
    def __init__(self, norm=None):
        super(Fft, self).__init__()
        self.norm = norm

    def forward(self, X_re, X_im):
        X_re, X_im = make_contiguous(X_re, X_im)
        k_re, k_im = fft(X_re, X_im)

        if self.norm == 'ortho':
            N = np.sqrt(k_re.size(-1))
            k_re /= N
            k_im /= N
        return k_re, k_im

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re,
                                                         grad_output_im)
        gi, gr = fft(grad_output_im,grad_output_re)

        if self.norm == 'ortho':
            N = np.sqrt(gi.size(-1))
            gi /= N
            gr /= N

        return gr,gi


class Ifft(torch.autograd.Function):
    def __init__(self, norm=None):
        super(Ifft, self).__init__()
        self.norm = norm

    def forward(self, k_re, k_im):
        k_re, k_im = make_contiguous(k_re, k_im)
        x_re, x_im = ifft(k_re, k_im)

        if self.norm == 'ortho':
            N = np.sqrt(x_re.size(-1))
            x_re *= N
            x_im *= N

        return x_re, x_im

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re,
                                                         grad_output_im)
        gi, gr = ifft(grad_output_im,grad_output_re)

        if self.norm == 'ortho':
            N = np.sqrt(gi.size(-1))
            gi *= N
            gr *= N

        return gr, gi


class Fft2d(torch.autograd.Function):
    def __init__(self, norm=None):
        super(Fft2d, self).__init__()
        self.norm = norm

    def forward(self, X_re, X_im):
        X_re, X_im = make_contiguous(X_re, X_im)
        k_re, k_im = fft2(X_re, X_im)
        if self.norm == 'ortho':
            N = np.sqrt(k_re.size(-1) * k_re.size(-2))
            k_re /= N
            k_im /= N
        return k_re, k_im

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re,
                                                         grad_output_im)
        gi, gr = fft2(grad_output_im,grad_output_re)

        if self.norm == 'ortho':
            N = np.sqrt(gi.size(-1) * gi.size(-2))
            gi /= N
            gr /= N

        return gr, gi


class Ifft2d(torch.autograd.Function):
    def __init__(self, norm=None):
        super(Ifft2d, self).__init__()
        self.norm = norm

    def forward(self, k_re, k_im):
        k_re, k_im = make_contiguous(k_re, k_im)
        x_r, x_i = ifft2(k_re, k_im)
        if self.norm == 'ortho':
            N = np.sqrt(x_r.size(-1) * x_r.size(-2))
            x_r *= N
            x_i *= N
        return x_r, x_i

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re,
                                                         grad_output_im)
        gi, gr = ifft2(grad_output_im,grad_output_re)
        if self.norm == 'ortho':
            N = np.sqrt(gi.size(-1) * gi.size(-2))
            gi *= N
            gr *= N

        return gr, gi


def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + k0
    return out


class DataConsistencyInKspace(object):
    """ Create data consistency operator """

    def __init__(self, noise_lvl=None, norm='ortho'):
        self.fft2_fun = Fft2d(norm)
        self.ifft2_fun = Ifft2d(norm)
        self.noise_lvl = noise_lvl

    def perform(self, x, k0, mask):
        """
        x    - input in k-space
        x0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        k = torch.cat(self.fft2_fun(x[:, 0:1], x[:, 1:2]),1)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = torch.cat(self.ifft2_fun(out[:, 0:1], out[:, 1:2]),1)

        return x_res


if __name__ == '__main__':

    from torch.autograd import Variable
    import scipy.io as sio

    NORM='ortho' # 'ortho'
    seq = sio.loadmat('/vol/medic02/users/js3611/cardiac_mri/jose/cardiac_complex/matlab/res_data_tmi_10_ac4.mat')['seq']

    seq_r, seq_i = np.real(seq), np.imag(seq)
    seq_r, seq_i = torch.from_numpy(seq_r).cuda(), torch.from_numpy(seq_i).cuda()
    seq_r, seq_i = Variable(seq_r, requires_grad=True), Variable(seq_i,requires_grad=True)

    fft_fn = Fft()
    ifft_fn= Ifft()
    k_r, k_i = fft_fn(seq_r, seq_i)
    x_r, x_i = ifft_fn(k_r, k_i)

    x = x_r.data.cpu().numpy() + x_i.data.cpu().numpy() * 1j
    k = k_r.data.cpu().numpy() + k_i.data.cpu().numpy() * 1j

    print('FFT/IFFT')
    print(np.allclose(seq, x))
    print(np.allclose(np.fft.fft(seq), k))
    print(np.allclose(seq, np.fft.ifft(k)))

    seq_r, seq_i = np.real(seq), np.imag(seq)
    seq_r, seq_i = torch.from_numpy(seq_r).cuda(), torch.from_numpy(seq_i).cuda()
    seq_r, seq_i = Variable(seq_r, requires_grad=True), Variable(seq_i,requires_grad=True)

    fft_ofn = Fft(NORM)
    ifft_ofn= Ifft(NORM)
    k_r, k_i = fft_ofn(seq_r, seq_i)
    x_r, x_i = ifft_ofn(k_r, k_i)

    x = x_r.data.cpu().numpy() + x_i.data.cpu().numpy() * 1j
    k = k_r.data.cpu().numpy() + k_i.data.cpu().numpy() * 1j

    print('FFT/IFFT (ortho)')
    print(np.allclose(seq, x))
    print(np.allclose(np.fft.fft(seq, norm=NORM), k))
    print(np.allclose(seq, np.fft.ifft(k, norm=NORM)))
    print(np.allclose(np.fft.ifft(np.fft.fft(seq, norm=NORM),norm=NORM), x))

    # FFT2

    seq_r, seq_i = np.real(seq), np.imag(seq)
    seq_r, seq_i = torch.from_numpy(seq_r).cuda(), torch.from_numpy(seq_i).cuda()
    seq_r, seq_i = Variable(seq_r, requires_grad=True), Variable(seq_i,requires_grad=True)

    fft2_fn = Fft2d()
    ifft2_fn= Ifft2d()
    k_r, k_i = fft2_fn(seq_r, seq_i)
    x_r, x_i = ifft2_fn(k_r, k_i)

    x = x_r.data.cpu().numpy() + x_i.data.cpu().numpy() * 1j
    k = k_r.data.cpu().numpy() + k_i.data.cpu().numpy() * 1j

    print('FFT2/IFFT2')
    print(np.allclose(seq, x))
    print(np.allclose(np.fft.fft2(seq), k))

    seq_r, seq_i = np.real(seq), np.imag(seq)
    seq_r, seq_i = torch.from_numpy(seq_r).cuda(), torch.from_numpy(seq_i).cuda()
    seq_r, seq_i = Variable(seq_r, requires_grad=True), Variable(seq_i,requires_grad=True)

    fft2_ofn = Fft2d(NORM)
    ifft2_ofn= Ifft2d(NORM)
    k_r, k_i = fft2_ofn(seq_r, seq_i)
    x_r, x_i = ifft2_ofn(k_r, k_i)

    x = x_r.data.cpu().numpy() + x_i.data.cpu().numpy() * 1j
    k = k_r.data.cpu().numpy() + k_i.data.cpu().numpy() * 1j

    print('FFT2/IFFT2 (ortho)')
    print(np.allclose(seq, x))
    print(np.allclose(np.fft.fft2(seq, norm=NORM), k))
    print(np.allclose(seq, np.fft.ifft2(k, norm=NORM)))
    print(np.allclose(np.fft.ifft2(np.fft.fft2(seq, norm=NORM),norm=NORM), x))


    def create_complex_var(*args):
        return (torch.autograd.Variable(torch.randn(*args).double().cuda(), requires_grad=True),
                torch.autograd.Variable(torch.randn(*args).double().cuda(), requires_grad=True))

    def test_fft_gradcheck():
        invar = create_complex_var(3,5,5)
        assert torch.autograd.gradcheck(Fft(NORM), invar)

    def test_ifft_gradcheck():
        invar = create_complex_var(3,5,5)
        assert torch.autograd.gradcheck(Ifft(NORM), invar)

    def test_fft2d_gradcheck():
        invar = create_complex_var(5,3,3,3)
        assert torch.autograd.gradcheck(Fft2d(NORM), invar)

    def test_ifft2d_gradcheck():
        invar = create_complex_var(5,3,3,3)
        assert torch.autograd.gradcheck(Ifft2d(NORM), invar)

    res = test_fft_gradcheck()
    res = test_ifft_gradcheck()
    res = test_fft2d_gradcheck()
    res = test_ifft2d_gradcheck()

