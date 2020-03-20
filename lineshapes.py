#! /usr/bin/env python

import numpy as np
from scipy import stats, signal, interpolate

def RelativisticBreitWigner(s, m, g):
    """ Relativistic Breit-Wigner """
    return (m**2 - s - 1j*m*g)**-1

def Norm(x, y):
    """ """
    return np.sum(y) * (x[1]-x[0]) * x.shape[0]

def Normed(x, y):
    """  """
    return y / Norm(x, y)

def cohsum(f, g, frac, phi):
    """ """
    return (1-frac)*f + frac*g*np.exp(1j*phi)

def convolve(x, s, sigma):
    """ Discrete convolution with Gaussian resolution """
    return np.dot(s, stats.norm.pdf(x.reshape(1,-1) - x.reshape(-1,1), scale=sigma))

# def resolution(x, sigma, smax=5):
#     """ Gaussian resolution function """
#     return stats.norm.pdf(np.linspace(-smax*sigma, smax*sigma, x.shape[0]), scale=sigma)

def model(x, mass, width, fcoh, fbkg, phase, sigma, b, bcoh):
    """ Args:
         mass: mass of the resonance
        width: width of the resonance
         fcoh: fraction of coherent background
         fbkg: fraction of non-coherent background
        phase: phase of the coherent background
        sigma: width of the Gaussian resolution
    """
    x0 = (x - x[0]) / (x[-1] - x[0])
    pcoh = Normed(x, np.polynomial.polynomial.polyval(x0, bcoh))
    rbw = RelativisticBreitWigner(x**2, mass, width)
    rbw /= Norm(x, np.abs(rbw)**2)
    pbkg = np.polynomial.polynomial.polyval(x0, b)
    s = np.abs(cohsum(rbw, pcoh, fcoh, phase))**2
    if sigma > 0:
        s = convolve(x, s, sigma)
    sig, bkg = (1. - fbkg)*Normed(x, s), fbkg*Normed(x, pbkg)
    return (sig, bkg, sig + bkg)

def make_pdf(lo, hi, params):
    """ """
    grid = 500
    x = np.linspace(lo, hi, grid)
    # x = np.linspace(0.99*lo, 1.01*hi, grid)
    _, _, total = model(x, **params)
    # x, total = [a[int(0.1*grid):int(0.9*grid)] for a in [x, total]]
    assert(x[0] <= lo and x[-1] >= hi)
    # print(f'lo {lo} -> {x[0]}, hi {hi} -> {x[-1]}')
    return (interpolate.CubicSpline(x, total), 1.01 * max(total))

def params():
    """ Model parameters """
    return {
         'mass': 3.872,
        'width': 0.001,
         'fcoh': 0.01,
         'fbkg': 0.3,
        'phase': 0,
        'sigma': 0.00243,
         'bcoh': [1, 1],
            'b': [1, 0, 2]
    }
