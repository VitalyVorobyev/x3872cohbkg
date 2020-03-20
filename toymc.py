#! /usr/bin/env python

import numpy as np

from lineshapes import make_pdf, params

def accept_reject(N, lo, hi, pdf, maj, chunk=10**6):
    """ """
    events = []
    while len(events) < N:
        print(f'next chunk {len(events)}')
        rndm = np.random.rand(chunk, 2)
        x = lo + (hi-lo)*rndm[:,0]
        xi = maj * rndm[:,1]
        events += list(x[xi < pdf(x)])
    return events[:N]

def generate(N, pdict=params()):
    """ """
    lo, hi = 3.85, 3.90
    pdf, maj = make_pdf(lo, hi, pdict)
    return accept_reject(N, lo, hi, pdf, maj)

def main(N):
    """ """
    np.save('events', generate(N))

if __name__ == '__main__':
    main(10**5)
