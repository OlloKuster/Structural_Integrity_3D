import jax.numpy as np
from jax.scipy.signal import convolve
import dm_pix as pix
import scipy.ndimage as ndimage


def f2param(x, lims):
    """
    Scales x from [0, 1] to [lims[0], lims[1]].
    :param x: Density distribution to be scaled.
    :param lims: Limits of the scaled distribution.
    :return: Rescaled array ranging from lims[0] to lims[1].
    """
    (a, b) = lims
    return (b - a) * x + a

def f2gauss(x, sigma=0.01):
    """
    Uses dm_pix to apply a Gaussian filter to x with standard deviation sigma.
    :param x: The array where the Gaussian blur will be applied to.
    :param sigma: Standard deviation of the Gaussian blur.
    :return: Gaussian blurred array.
    """

    def gkernel(sigma):
        l = int(2 * np.ceil(4.0 * sigma) + 1)
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        xx, yy, zz = np.meshgrid(ax, ax, ax)

        kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2 + zz**2) / sigma ** 2)
        return kernel / np.sum(kernel)

    kernel = gkernel(sigma)
    res = convolve(x, kernel, mode='same')
    return res

def f2gauss_pix(x, sigma=0.01):
    """
    Uses dm_pix to apply a Gaussian filter to x with standard deviation sigma.
    :param x: The array where the Gaussian blur will be applied to.
    :param sigma: Standard deviation of the Gaussian blur.
    :return: Gaussian blurred array.
    """
    sigma = np.float64(sigma)
    kernel_size = int(2 * np.ceil(4.0 * sigma) + 1)
    filt = pix.gaussian_blur(x, sigma, kernel_size, padding='SAME')
    return filt


def f2bin(x, alpha=30, beta=0.5):
    """
    Binarises the values of x with parameters alpha and beta.
    :param x: Array which will be binarised.
    :param alpha: Steepness of the binarisation function.
    :param beta: Origin of the binarisation function.
    :return: Binarised array of x.
    """
    num = np.tanh(alpha * beta) + np.tanh(alpha * (x - beta))
    denom = np.tanh(alpha * beta) + np.tanh(alpha * (1 - beta))
    proj = num / denom
    return proj
