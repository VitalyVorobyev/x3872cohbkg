#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from lineshapes import model, params, make_pdf

def show_model(pdict=params()):
    N = 1000
    x0 = np.linspace(3.84, 3.91, N)

    s, b, total = model(x0, **pdict)
    fig, _ = plt.subplots(figsize=(7,8))
    tot, = plt.plot(x0, total)
    sig, = plt.plot(x0, s)
    bkg, = plt.plot(x0, b)
    plt.ylim(0., 1.5*max(total))
    plt.xlim(3.85, 3.9)
    plt.grid()
    plt.tight_layout()

    plt.subplots_adjust(left=0.1, bottom=0.25)

    axcolor = 'lightgoldenrodyellow'
    axphi  = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    axfcoh = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
    axfbkg = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    sphi  = Slider( axphi,  'phi', -np.pi, np.pi, valinit=pdict['phase'], valfmt='%1.3f')
    sfcoh = Slider(axfcoh, 'fcoh',     0.,  0.05, valinit=pdict['fcoh'],  valfmt='%1.3f')
    sfbkg = Slider(axfbkg, 'fbkg',     0.,  1.00, valinit=pdict['fbkg'],  valfmt='%1.3f')

    def update(val):
        pdict.update({
            'phase': sphi.val,
            'fcoh' : sfcoh.val,
            'fbkg' : sfbkg.val
        })
        s, b, total = model(x0, **pdict)
        tot.set_ydata(total)
        sig.set_ydata(s)
        bkg.set_ydata(b)
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
