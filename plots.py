#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from lineshapes import model, params, make_pdf

def make_plot(x, y):
    """ """
    y /= np.sum(y) * (x[1] - x[0]) * x.shape[0]
    plt.figure(figsize=(8,6))
    plt.plot(x, y)
    plt.ylim(0., 1.05 * max(y))
    plt.grid()
    plt.tight_layout()

def show_bw(pdict=params()):
    pdict=params()
    pdict['sigma'] = -1
    pdict['fcoh'] = 0
    pdict['fbkg'] = 0

    N = 500
    x = np.linspace(3.85, 3.9, N)
    y = model(x, **pdict)
    make_plot(x, y)

def show_bw_conv(pdict=params()):
    pdict=params()
    pdict['fcoh'] = 0
    pdict['fbkg'] = 0

    N = 500
    x = np.linspace(3.85, 3.9, N)
    y = model(x, **pdict)
    make_plot(x, y)

def show_model(pdict=params()):
    N = 500
    x = np.linspace(3.84, 3.91, N)
    s, b, total = model(x, **pdict)
    
    plt.figure(figsize=(8,6))
    plt.plot(x, total)
    plt.plot(x, s)
    plt.plot(x, b)
    plt.ylim(0., 1.05 * max(total))
    plt.xlim(3.85, 3.9)
    plt.grid()
    plt.tight_layout()

def show_data(fname):
    """ """
    data = np.load(fname)
    plt.figure(figsize=(8,6))
    plt.hist(data, bins=250)
    plt.grid()
    plt.tight_layout()

def show_fit(data, pars, weight=None):
    """ """
    dots, bins, N = 500, 250, data.shape[0]
    if weight is None:
        weight = np.ones(data.shape)
    lo, hi = min(data), max(data)
    pdf, _ = make_pdf(lo, hi, pars)
    x = np.linspace(lo, hi, dots)
    y = pdf(x)
    integral_pdf = np.sum(y) / dots
    integral_hist = np.sum(weight) / bins
    y /= integral_pdf / integral_hist
    binsize = (hi - lo) / bins * 10**3

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.figure(figsize=(8,6))
    plt.hist(data, bins=bins, weights=weight)
    plt.ylabel('events / {:.2f} MeV'.format(binsize), fontsize=16)
    plt.xlabel(r'$m(J/\psi\pi^+\pi^-)$, GeV'.format(binsize), fontsize=16)
    plt.plot(x, y)
    plt.grid()
    plt.tight_layout()
    plt.show()

def show_hist_fit(bins, hdata, herrs, pars):
    """ """
    lo, hi = bins[0], bins[-1]
    dots, nbins, N = 500, hdata.shape[0], np.sum(hdata)
    pdf, spdf, bpdf, _ = make_pdf(lo, hi, pars)
    x = np.linspace(lo, hi, dots)
    y = pdf(x)
    norm = np.sum(y) / dots / (N / nbins)
    y /= norm
    binsize = (hi - lo) / nbins * 10**3

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.figure(figsize=(8,6))
    plt.errorbar(bins, hdata, yerr=herrs, linestyle='none', marker='.', markersize=2)
    plt.ylabel('events / {:.2f} MeV'.format(binsize), fontsize=16)
    plt.xlabel(r'$m(J/\psi\pi^+\pi^-)$, GeV'.format(binsize), fontsize=16)
    plt.plot(x, y)
    plt.plot(x, bpdf(x) / norm)
    plt.xlim(3.85, 3.90)
    plt.ylim(0.00, 250)
    # plt.fill_between(x, spdf(x) / norm, 0)
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    """ """
    # show_bw()
    # show_bw_conv()
    # show_resolution()
    show_model()
    # show_data('events.npy')
    plt.show()

if __name__ == '__main__':
    main()
