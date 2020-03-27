#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import sys
import scipy.integrate as integrate
from iminuit import Minuit

from lineshapes import *
from plots import show_fit, show_hist_fit

def convert_dataset():
    """ """
    import ROOT as r
    f = r.TFile('~/dataset.root','r')
    ds = f.dataset
    print(ds.sumEntries())
    ds.get(1).Print()

    events = []

    for i in range(int(ds.sumEntries())):
        aset = ds.get(i)
        events.append([
            aset.getRealValue('B_b_sw'),
            aset.getRealValue('S_b_sw'),
            aset.getRealValue('m134BC')
        ])
    np.save('dataset', events)

def show_dataset():
    """ """
    events = np.load('dataset.npy')
    m, ws, wb = events[:,2], events[:,1], events[:,0]
    print(m)
    print(ws)
    print(wb)

    plt.hist(m, bins=150)
    plt.hist(m, bins=150, weights=ws)
    plt.show()

class Fitter(object):
    def fcn(self, mass, width, fcoh, fbkg, phase, sigma, b0, b1):
        self.pars = {
             'mass': mass,
            'width': width,
             'fcoh': fcoh,
             'fbkg': fbkg,
            'phase': phase,
            'sigma': sigma,
             'bcoh': [1, b0, b1],
                'b': [1, b0, b1]
        }
        self.pdf = make_pdf(self.lo, self.hi, self.pars)[0]
        self.norm, _ = integrate.quad(self.pdf, self.lo, self.hi)

        loglh = self.loglh()
        print('loglh: {:.2f}, m {:.3f}, w {:.3f}, fcoh {:.3f}, fbck {:.3f}, phi {:.3f}, norm {:.3f}'.format(
            loglh, mass*10**3, width*10**3, fcoh, fbkg, phase, self.norm))
        return loglh
    
    def loglh(self):
        return -np.dot(self.weights, np.log(self.pdf(self.data))) + np.log(self.norm) * self.data.shape[0]

    def fitTo(self, data, weights, init):
        self.data = data
        self.weights = weights
        self.lo, self.hi = min(data), max(data)
        
        self.minimizer = Minuit(self.fcn, errordef=0.5, **init) 

        fmin, param = self.minimizer.migrad()
        self.minimizer.print_param()
        corrmtx = self.minimizer.matrix(correlation=True)
        return (fmin, param, corrmtx)

class FitterBinned(object):
    def fcn(self, mass, width, fcoh, fbkg, phase, sigma, bcoh, bbkg):
        self.pars = {
             'mass': mass, 'width': width,
             'fcoh': fcoh, 'fbkg': fbkg,
            'phase': phase, 'sigma': sigma,
             'bcoh': [1, bcoh],
                'b': [1, bbkg]
            }
        self.hpdf = make_pdf_hist(self.bins, self.pars, self.nevt)
        chisq = self.chisq()
        print('chisq: {:.2f}, m {:.3f}, w {:.3f}, fcoh {:.3f}, fbck {:.3f}, phi {:.3f} bcoh {:.3f} bbkg {:.3f}'.format(
            chisq, mass*10**3, width*10**3, fcoh, fbkg, phase, bcoh, bbkg))
        return chisq

    def chisq(self):
        return np.sum(((self.data - self.hpdf) / self.errs)**2)

    def fitTo(self, bins, hdata, init):
        self.data, self.errs = hdata, np.sqrt(hdata)
        self.bins, self.nevt = bins, np.sum(hdata)
        self.minimizer = Minuit(self.fcn, errordef=0.5, **init) 

        fmin, param = self.minimizer.migrad()
        self.minimizer.print_param()
        corrmtx = self.minimizer.matrix(correlation=True)
        return (fmin, param, corrmtx)

def rndm_angle():
    return -np.pi + 2.*np.pi*np.random.random()

def mnpardict(name, val, err, range, fixed):
    return {name: val, f'error_{name}': err, f'limit_{name}': range, f'fix_{name}': fixed}

def combine_dicts(dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res

def init_full_fit(pars=params()):
    """ """
    return combine_dicts([
        mnpardict( 'mass', pars['mass'],       0.01,   (3.86, 3.89),    False),
        mnpardict('width', pars['width'],      0.0005, (0., 0.0025),    False),
        mnpardict( 'fcoh', np.random.random(), 0.1,    (0., 1.),        False),
        mnpardict( 'fbkg', np.random.random(), 0.1,    (0., 1.),        False),
        mnpardict('phase', rndm_angle(),       0.1,    (-np.pi, np.pi), False),
        mnpardict('sigma', pars['sigma'],      0.1,    (0.0001, 0.005), True),
        mnpardict( 'bcoh', 10**3,              10,     (-100., 10**4.), False),
        mnpardict( 'bbkg', 0,                  10,     (-100., 10**4.), False),
    ])

def init_noncoh_fit(pars=params()):
    """ """
    return combine_dicts([
        mnpardict( 'mass', pars['mass'],       0.01,   (3.86, 3.89),    False),
        mnpardict('width', pars['width'],      0.0005, (0., 0.0025),    False),
        mnpardict( 'fcoh', 0,                  0.1,    (0., 1.),        True),
        mnpardict( 'fbkg', np.random.random(), 0.1,    (0., 1.),        False),
        mnpardict('phase', rndm_angle(),       0.1,    (-np.pi, np.pi), True),
        mnpardict('sigma', pars['sigma'],      0.1,    (0.0001, 0.005), True),
        mnpardict( 'bcoh', 10**3,              10,     (-100., 10**4.), True),
        mnpardict( 'bbkg', 0,                  10,     (-100., 10**4.),   False),
    ])

def init_coh_fit(pars=params()):
    """ """
    return combine_dicts([
        mnpardict( 'mass', pars['mass'],       0.01,   (3.86, 3.89),    False),
        mnpardict('width', pars['width'],      0.0005, (0., 0.0025),    False),
        mnpardict( 'fcoh', np.random.random(), 0.1,    (0., 1.),        False),
        mnpardict( 'fbkg', 0,                  0.1,    (0., 1.),        True),
        mnpardict('phase', rndm_angle(),       0.1,    (-np.pi, np.pi), False),
        mnpardict('sigma', pars['sigma'],      0.1,    (0.0001, 0.005), True),
        mnpardict( 'bcoh', 10**3,              10,     (-100., 10**4.), False),
        mnpardict( 'bbkg', 0,                  10,     (-100., 10**4.),   True),
    ])

def make_hist(events, weights, bins=150, range=[3.85, 3.90]):
    hist, bins = np.histogram(events, bins=bins, range=range, weights=weights)
    bins = 0.5*(bins[1:]+bins[:-1])
    return (bins, hist, np.sqrt(hist))

def binned_fit(events, init, weights=None, bins=150):
    if weights is None:
        weights = np.ones(events.shape)
    bins, hdata, herrs = make_hist(events, weights, bins)
    assert(bins.shape == hdata.shape)
    f = FitterBinned()
    fmin, par, corrmtx = f.fitTo(bins, hdata, init())
    print(fmin)
    print(corrmtx)

    show_hist_fit(bins, hdata, herrs, f.pars)
    return (fmin, par, corrmtx, f.pars)

def unbinned_fit(events, init, weights=None):
    f = Fitter()
    if weights is None:
        weights = np.ones(events.shape)
    fmin, par, corrmtx = f.fitTo(events, weights, init())
    show_fit(events, f.pars, weights)
    return (fmin, par, corrmtx, f.pars)

def main():
    """ """
    events = np.load('dataset.npy')
    m, ws, wb = events[:,2], events[:,1], events[:,0]
    print(ws.sum(), m.shape[0])

    if False:
        ws = np.ones(ws.shape)
        fmin, par, corrmtx, pardict = unbinned_fit(m, init_full_fit, ws)
    else:
        # ws = np.ones(ws.shape)
        fmin, par, corrmtx, pardict = binned_fit(m, init_full_fit, ws)
    
    print(' mass: {:>7.2f} +- {:>4.2f} MeV'.format(par[0]['value']*10**3, par[0]['error']*10**3))
    print('width: {:>7.2f} +- {:>4.2f} MeV'.format(par[1]['value']*10**3, par[1]['error']*10**3))

if __name__ == '__main__':
    main()
