#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from lineshapes import model, params, make_pdf

def show_model(pdict=params()):
    N = 1000
    x = np.linspace(3.84, 3.91, N)
    # lo, hi = 0.25, 0.5
    lo, hi = 0., 1.

    s, b, total = model(x, **pdict)
    x0 = x
    # x0, s, b, total = [a[int(lo*N):int(hi*N)] for a in [x, s, b, total]]
    # plt.figure(figsize=(8,6))
    fig, ax = plt.subplots(figsize=(7,8))
    tot, = plt.plot(x0, total)
    sig, = plt.plot(x0, s)
    bkg, = plt.plot(x0, b)
    plt.ylim(0., 0.10)
    plt.xlim(3.85, 3.9)
    plt.grid()
    plt.tight_layout()

    plt.subplots_adjust(left=0.1, bottom=0.25)

    axcolor = 'lightgoldenrodyellow'
    axphi  = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axfcoh = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    axfbkg = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    sphi  = Slider( axphi,  'phi', -np.pi, np.pi, valinit=0.)
    sfcoh = Slider(axfcoh, 'fcoh', 0, 0.05, valinit=0.01)
    sfbkg = Slider(axfbkg, 'fbkg', 0., 1., valinit=0.3)

    def update(val):
        pdict['phase'] = sphi.val
        pdict['fcoh'] = sfcoh.val
        pdict['fbkg'] = sfbkg.val
        # s, b, total = [a[int(lo*N):int(hi*N)] for a in model(x, **pdict)]
        s, b, total = model(x, **pdict)
        tot.set_ydata(total)
        sig.set_ydata(s)
        bkg.set_ydata(b)
        # ax.ylim(0., 1.05 * max(total))
        fig.canvas.draw_idle()

    sphi.on_changed(update)
    sfcoh.on_changed(update)
    sfbkg.on_changed(update)
    plt.show()

def main():
    """ """
    show_model()

if __name__ == '__main__':
    main()
