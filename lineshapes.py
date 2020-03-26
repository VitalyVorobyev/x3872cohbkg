#! /usr/bin/env python

import numpy as np
from scipy import stats, signal, interpolate


# def crystalball(x, mean, sigma, x0, alpha, n):
#     """ Crystal Ball shape with left tail """
#     result = np.empty(x.shape)
#     a = (n / np.abs(alpha))**n * np.exp(-0.5*alpha*2)
#     b = n / np.abs(alpha) - np.abs(alpha)

def resolution(x, scale, n = 20, alpha = 1.485):
    xi = x / scale
    return 0.5 * (stats.crystalball.pdf( xi, n, alpha)+
                  stats.crystalball.pdf(-xi, n, alpha))

def RelativisticBreitWigner(s, m, g):
    """ Relativistic Breit-Wigner """
    return (m**2 - s - 1j*m*g)**-1

def Norm(x, y, window=None):
    """ """
    if window is not None:
        mask = (x > window[0]) & (x < window[1])
        return np.sum(y[mask]) # * (x[1]-x[0]) * x[mask].shape[0]
    return np.sum(y) # * (x[1]-x[0]) * x.shape[0]

def Normed(x, y, window=None, transform=None):
    """  """
    if transform is not None:
        return y / Norm(x, transform(y), window)
    return y / Norm(x, y, window)

def cohsum(f, g, frac, phi):
    """ """
    return (1-frac)*f + frac*g*np.exp(1j*phi)

def convolve(x, s, sigma, pdf=stats.norm.pdf):
    """ Discrete convolution with Gaussian resolution """
    return np.dot(s, pdf(x.reshape(1,-1) - x.reshape(-1,1), scale=sigma))

def model(x, mass, width, fcoh, fbkg, phase, sigma, b, bcoh):
    """ Args:
         mass: mass of the resonance
        width: width of the resonance
         fcoh: fraction of coherent background
         fbkg: fraction of non-coherent background
        phase: phase of the coherent background
        sigma: width of the Gaussian resolution
    """
    norm_window = (mass - 3.*max(sigma, width),
                   mass + 3.*max(sigma, width))
    x0 = (x - x[0]) / (x[-1] - x[0])
    pcoh = Normed(x, np.polynomial.polynomial.polyval(x0, bcoh), norm_window, lambda x: x**2)
    rbw  = Normed(x, RelativisticBreitWigner(x**2, mass, width), norm_window, lambda x: np.abs(x)**2)
    pbkg = Normed(x, np.polynomial.polynomial.polyval(x0, b)   , norm_window)
    s = np.abs(cohsum(rbw, pcoh, fcoh, phase))**2
    if sigma > 0:
        s = convolve(x, s, sigma, pdf=resolution)
    sig, bkg = (1. - fbkg)*Normed(x, s, norm_window), fbkg*pbkg
    return (sig, bkg, sig + bkg)

def make_pdf(lo, hi, params):
    """ """
    grid = 500
    x = np.linspace(0.9*lo, 1.1*hi, grid)
    _, _, total = model(x, **params)
    return (interpolate.CubicSpline(x, total), 1.01 * max(total))

def make_pdf_hist(bins, params, nevt):
    """ """
    y = make_pdf(bins[0], bins[-1], params)[0](bins)
    return y / np.sum(y) * nevt

def params():
    """ Model parameters """
    return {
         'mass': 3.872,
        'width': 0.001,
         'fcoh': 0.008,
         'fbkg': 0.000,
        'phase': 1.57,
        'sigma': 0.00243,
         'bcoh': [1, 1],
            'b': [1, 1, 0]
    }

def main():
    import matplotlib.pyplot as plt
    x = np.linspace(-5, 5, 500)
    plt.plot(x, resolution(x, 2.43))
    plt.plot(x, stats.norm.pdf(x, scale=2.43))
    plt.show()

if __name__ == '__main__':
    main()
