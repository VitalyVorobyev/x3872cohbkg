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
    
    # lo, hi = int(0.08*N), int(0.92*N)
    # x, s, b, total = [a[lo:hi] for a in [x, s, b, total]]
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

def show_fit(data, pars):
    """ """
    dots, bins, N = 500, 250, data.shape[0]
    lo, hi = min(data), max(data)
    pdf, _ = make_pdf(lo, hi, pars)
    x = np.linspace(lo, hi, dots)
    y = pdf(x)
    # y /= np.sum(y) * (x[1] - x[0]) * bins / dots / 10
    y /= np.sum(y) * (x[1] - x[0]) * (x[-1] - x[0])

    plt.figure(figsize=(8,6))
    plt.hist(data, bins=bins)
    plt.plot(x, y)
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
