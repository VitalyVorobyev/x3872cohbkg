#! /usr/bin/env python

import sys
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
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
        # print('loglh: {:.2f}, m {:.3f}, w {:.3f}, fcoh {:.3f}, fbck {:.3f}, phi {:.3f}, norm {:.3f}'.format(
            # loglh, mass*10**3, width*10**3, fcoh, fbkg, phase, self.norm))
        return loglh
    
    def loglh(self):
        return -np.sum(np.log(self.pdf(self.data))) + np.log(self.norm) * self.data.shape[0]

    def fitTo(self, data, init):
        self.data = data
        self.lo, self.hi = min(data), max(data)
        
        self.minimizer = Minuit(self.fcn, errordef=0.5, **init) 

        fmin, param = self.minimizer.migrad()
        self.minimizer.print_param()
        corrmtx = self.minimizer.matrix(correlation=True)
        return (fmin, param, corrmtx)

def rndm_angle():
    return -np.pi + 2.*np.pi*np.random.random()

def init_full_fit(pars=params()):
    """ """
    return {
         'mass': pars['mass'],       'error_mass': 0.01,   'limit_mass': (3.86, 3.89),     'fix_mass': False,
        'width': pars['width'],     'error_width': 0.001, 'limit_width': (0., 0.002),     'fix_width': False,
         'fcoh': np.random.random(), 'error_fcoh': 0.1,    'limit_fcoh': (0., 1.),         'fix_fcoh': False,
         'fbkg': np.random.random(), 'error_fbkg': 0.1,    'limit_fbkg': (0., 1.),         'fix_fbkg': False,
        'phase': rndm_angle(),      'error_phase': 0.1,   'limit_phase': (-np.pi, np.pi), 'fix_phase': False,
        'sigma': pars['sigma'],     'error_sigma': 0.1,   'limit_sigma': (0.0001, 0.005), 'fix_sigma': True,
         'bcoh': pars['bcoh'][1],    'error_bcoh': 0.1,    'limit_bcoh': (0.5, 1.5),       'fix_bcoh': False,
           'b0': pars['b'][1],         'error_b0': 0.1,      'limit_b0': (-1., 1.),          'fix_b0': True,
           'b1': pars['b'][2],         'error_b1': 0.1,      'limit_b1': (-1., 3.),          'fix_b1': True
    }

def init_noncoh_fit(pars=params()):
    """ """
    return {
         'mass': pars['mass'],       'error_mass': 0.01,   'limit_mass': (3.86, 3.89),     'fix_mass': False,
        'width': pars['width'],     'error_width': 0.001, 'limit_width': (0., 0.002),     'fix_width': False,
         'fcoh': 0,                  'error_fcoh': 0.1,    'limit_fcoh': (0., 1.),         'fix_fcoh': True,
         'fbkg': np.random.random(), 'error_fbkg': 0.1,    'limit_fbkg': (0., 1.),         'fix_fbkg': False,
        'phase': pars['phase'],     'error_phase': 0.1,   'limit_phase': (-np.pi, np.pi), 'fix_phase': True,
        'sigma': pars['sigma'],     'error_sigma': 0.1,   'limit_sigma': (0.0001, 0.005), 'fix_sigma': True,
         'bcoh': pars['bcoh'][1],    'error_bcoh': 0.1,    'limit_bcoh': (0.5, 1.5),       'fix_bcoh': True,
           'b0': pars['b'][1],         'error_b0': 0.1,      'limit_b0': (-1., 1.),          'fix_b0': False,
           'b1': pars['b'][2],         'error_b1': 0.1,      'limit_b1': (-1., 3.),          'fix_b1': False
    }

def noncoh_phase_scan():
    """ """
    f = Fitter()
    pars = params()
    true_mass, true_width = [pars[key] for key in ['mass', 'width']]
    phi, fit_mass, fit_width, valid_fit = [], [], [], []
    with open('log1.txt', 'w') as file:
        file.write('Noncoherent fit with phase scan')
        for key, val in pars.items():
            file.write(f'{key}: {val}')
        for ph in np.linspace(-np.pi, np.pi, 20):
            file.write(f'phase {ph}')
            pars['phase'] = ph
            data = np.array(generate(10**5, pars))
            fmin, par, corrmtx = f.fitTo(data, init_noncoh_fit())
            phi.append(ph)
            fit_mass.append( [par[0]['value'], par[0]['error']])
            fit_width.append([par[1]['value'], par[1]['error']])
            valid_fit.append(fmin.is_valid)
            # show_fit(data, f.pars)
    fit_mass, fit_width = [np.array(x) for x in [fit_mass, fit_width]]
    print(valid_fit)
    print(fit_mass)
    print(fit_width)
    plt.figure()
    plt.errorbar(phi, fit_mass[:,0] - true_mass, yerr=fit_mass[:,1], marker='.', linestyle='none')
    plt.grid()
    plt.tight_layout()
    
    plt.figure()
    plt.errorbar(phi, fit_width[:,0] - true_width, yerr=fit_mass[:,1], marker='.', linestyle='none')
    plt.grid()
    plt.tight_layout()
    plt.show()

def uncoh_fit():
    f = Fitter()
    data = np.array(generate(10**5))
    fmin, par, corrmtx = f.fitTo(data, init_noncoh_fit())
    print(fmin)
    print(corrmtx)
    print(' mass: {:>7.2f} +- {:>4.2f} MeV'.format(par[0]['value']*10**3, par[0]['error']*10**3))
    print('width: {:>7.2f} +- {:>4.2f} MeV'.format(par[1]['value']*10**3, par[1]['error']*10**3))

    show_fit(data, f.pars)

def full_fit():
    f = Fitter()
    data = np.array(generate(10**5))
    fmin, par, corrmtx = f.fitTo(data, init_full_fit())
    print(fmin)
    print(corrmtx)
    print(' mass:  {:>7.2f} +- {:>4.2f} MeV'.format(par[0]['value']*10**3, par[0]['error']*10**3))
    print('width:  {:>7.2f} +- {:>4.2f} MeV'.format(par[1]['value']*10**3, par[1]['error']*10**3))
    print(' fcoh: ({:>7.2f} +- {:>4.2f}) / 100'.format(par[2]['value']*10**2, par[2]['error']*10**2))
    print(' fbkg: ({:>7.2f} +- {:>4.2f}) / 100'.format(par[3]['value']*10**2, par[3]['error']*10**2))
    print('phase:  {:>7.2f} +- {:>4.2f}'.format(par[4]['value'], par[4]['error']))

    show_fit(data, f.pars)

    bins, value, mres = f.minimizer.mnprofile('fbkg', bins=30, bound=2, subtract_min=False)
    print(bins)
    print(value)
    print(mres)

    scans1 = f.minimizer.mncontour('fcoh', 'fbkg', numpoints=20, sigma=1)
    print(scans1)
    for x,y, in scans1:
        print('  {:.3f}, {:.3f}'.format(x*100, y*100))

    x = [a[0] for a in scans1]
    y = [a[0] for a in scans1]
    plt.plot(x, y)
    plt.show()

def main():
    """ """
    if len(sys.argv) == 2:
        if sys.argv[1] == 'full':
            full_fit()
        elif sys.argv[1] == 'ucoh':
            uncoh_fit()
        elif sys.argv[1] == 'scan':
            noncoh_phase_scan()
        else:
            print('Wrong command')
            return
    else:
        uncoh_fit()
    
    # 
    # 
    # fmin, _, corrmtx = f.fitTo(data, init_full_fit())

    # print(fmin)
    # print(fmin.is_valid)
    # print(par[0]['error'])
    # show_fit(data, f.pars)

if __name__ == '__main__':
    main()
