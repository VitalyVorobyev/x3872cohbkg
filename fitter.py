#! /usr/bin/env python

import numpy as np
import scipy.integrate as integrate
from iminuit import Minuit

from lineshapes import *
from toymc import generate
from plots import show_fit

class Fitter(object):
    def fcn(self, mass, width, fcoh, fbkg, phase, sigma, bcoh, b0, b1):
        self.pars = {
             'mass': mass,
            'width': width,
             'fcoh': fcoh,
             'fbkg': fbkg,
            'phase': phase,
            'sigma': sigma,
             'bcoh': [1, bcoh],
                'b': [1, b0, b1]
        }
        self.pdf, _ = make_pdf(self.lo, self.hi, self.pars)
        self.norm, _ = integrate.quad(self.pdf, self.lo, self.hi)

        loglh = self.loglh()
        print('loglh: {:.2f}, m {:.3f}, w {:.3f}, fcoh {:.3f}, fbck {:.3f}, phi {:.3f}, norm {:.3f}'.format(
            loglh, mass*10**3, width*10**3, fcoh, fbkg, phase, self.norm))
        return loglh
    
    def loglh(self):
        return -np.sum(np.log(self.pdf(self.data))) + np.log(self.norm) * self.data.shape[0]

    def fitTo(self, data, init):
        self.data = data
        self.lo, self.hi = min(data), max(data)
        
        minimizer = Minuit(self.fcn, errordef=0.5, **init)

        fmin, param = minimizer.migrad()
        minimizer.print_param()
        corrmtx = minimizer.matrix(correlation=True)
        return (fmin, param, corrmtx)

def init_full_fit(pars=params()):
    """ """
    return {
         'mass': pars['mass'],     'error_mass': 0.01,   'limit_mass': (3.86, 3.89),     'fix_mass': False,
        'width': pars['width'],   'error_width': 0.001, 'limit_width': (0., 0.002),     'fix_width': False,
         'fcoh': pars['fcoh'],     'error_fcoh': 0.1,    'limit_fcoh': (0., 1.),         'fix_fcoh': False,
         'fbkg': pars['fbkg'],     'error_fbkg': 0.1,    'limit_fbkg': (0., 1.),         'fix_fbkg': False,
        'phase': pars['phase'],   'error_phase': 0.1,   'limit_phase': (-np.pi, np.pi), 'fix_phase': False,
        'sigma': pars['sigma'],   'error_sigma': 0.1,   'limit_sigma': (0.0001, 0.005), 'fix_sigma': True,
         'bcoh': pars['bcoh'][1],  'error_bcoh': 0.1,    'limit_bcoh': (0.5, 1.5),       'fix_bcoh': False,
           'b0': pars['b'][1],       'error_b0': 0.1,      'limit_b0': (-1., 1.),          'fix_b0': True,
           'b1': pars['b'][2],       'error_b1': 0.1,      'limit_b1': (-1., 3.),          'fix_b1': True
    }

def init_fixed_bck_fit(pars=params()):
    """ """
    return {
         'mass': pars['mass'],     'error_mass': 0.01,   'limit_mass': (3.86, 3.89),     'fix_mass': False,
        'width': pars['width'],   'error_width': 0.001, 'limit_width': (0., 0.002),     'fix_width': True,
         'fcoh': pars['fcoh'],     'error_fcoh': 0.1,    'limit_fcoh': (0., 1.),         'fix_fcoh': True,
         'fbkg': pars['fbkg'],     'error_fbkg': 0.1,    'limit_fbkg': (0., 1.),         'fix_fbkg': True,
        'phase': pars['phase'],   'error_phase': 0.1,   'limit_phase': (-np.pi, np.pi), 'fix_phase': True,
        'sigma': pars['sigma'],   'error_sigma': 0.1,   'limit_sigma': (0.0001, 0.005), 'fix_sigma': True,
         'bcoh': pars['bcoh'][1],  'error_bcoh': 0.1,    'limit_bcoh': (0.5, 1.5),       'fix_bcoh': True,
           'b0': pars['b'][1],       'error_b0': 0.1,      'limit_b0': (-1., 1.),          'fix_b0': True,
           'b1': pars['b'][2],       'error_b1': 0.1,      'limit_b1': (-1., 3.),          'fix_b1': True
    }

def init_noncoh_fit(pars=params()):
    """ """
    return {
         'mass': pars['mass'],     'error_mass': 0.01,   'limit_mass': (3.86, 3.89),     'fix_mass': False,
        'width': pars['width'],   'error_width': 0.001, 'limit_width': (0., 0.002),     'fix_width': False,
         'fcoh': 0,                'error_fcoh': 0.1,    'limit_fcoh': (0., 1.),         'fix_fcoh': True,
         'fbkg': pars['fbkg'],     'error_fbkg': 0.1,    'limit_fbkg': (0., 1.),         'fix_fbkg': False,
        'phase': pars['phase'],   'error_phase': 0.1,   'limit_phase': (-np.pi, np.pi), 'fix_phase': True,
        'sigma': pars['sigma'],   'error_sigma': 0.1,   'limit_sigma': (0.0001, 0.005), 'fix_sigma': True,
         'bcoh': pars['bcoh'][1],  'error_bcoh': 0.1,    'limit_bcoh': (0.5, 1.5),       'fix_bcoh': True,
           'b0': pars['b'][1],       'error_b0': 0.1,      'limit_b0': (-1., 1.),          'fix_b0': False,
           'b1': pars['b'][2],       'error_b1': 0.1,      'limit_b1': (-1., 3.),          'fix_b1': False
    }

def main():
    """ """
    f = Fitter()
    # data = np.load('events.npy')
    data = np.array(generate(10**5))
    # fmin, _, corrmtx = f.fitTo(data, init_fixed_bck_fit())
    fmin, _, corrmtx = f.fitTo(data, init_full_fit())
    print(fmin)
    show_fit(data, f.pars)
    fmin, _, corrmtx = f.fitTo(data, init_noncoh_fit())
    # print(corrmtx)
    print(fmin)
    show_fit(data, f.pars)

if __name__ == '__main__':
    main()
