#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from lineshapes import model, params, make_pdf

def show_model(pdict1=params(), pdict2=params()):
    N = 1000
    x0 = np.linspace(3.84, 3.91, N)

    pdict1['fcoh'] = 0
    pdict2['fbkg'] = 0

    fig, _ = plt.subplots(figsize=(9,8))

    s1, b1, total1 = model(x0, **pdict1)
    tot1, = plt.plot(x0, total1)
    # sig1, = plt.plot(x0, s1)
    # bkg1, = plt.plot(x0, b1)

    s2, b2, total2 = model(x0, **pdict2)
    tot2, = plt.plot(x0, total2)
    # sig2, = plt.plot(x0, s2)
    # bkg2, = plt.plot(x0, b2)
    
    plt.ylim(0., 1.5*max(total1))
    plt.xlim(3.85, 3.9)
    plt.grid()
    plt.tight_layout()

    plt.subplots_adjust(left=0.1, bottom=0.25)

    axcolor = 'lightgoldenrodyellow'
    axb2    = plt.axes([0.10, 0.17, 0.3, 0.03], facecolor=axcolor)
    axb1    = plt.axes([0.10, 0.12, 0.3, 0.03], facecolor=axcolor)
    axfbkg  = plt.axes([0.10, 0.07, 0.3, 0.03], facecolor=axcolor)
    axmass  = plt.axes([0.10, 0.02, 0.3, 0.03], facecolor=axcolor)

    sb2   = Slider(axb2,   'b2',  -5, 5, valinit=pdict1['b'][2], valfmt='%1.3f')
    sb1   = Slider(axb1,   'b1',  -5, 5, valinit=pdict1['b'][1], valfmt='%1.3f')
    sfbkg = Slider(axfbkg, 'fbkg', 0, 1, valinit=pdict1['fbkg'], valfmt='%1.3f')
    smass = Slider(axmass, 'mass', 3.868, 3.876, valinit=pdict1['mass'], valfmt='%1.3f')

    axphi  = plt.axes([0.55, 0.15, 0.3, 0.03], facecolor=axcolor)
    axfcoh = plt.axes([0.55, 0.10, 0.3, 0.03], facecolor=axcolor)
    axwdth = plt.axes([0.55, 0.05, 0.3, 0.03], facecolor=axcolor)

    sphi   = Slider(axphi,  'phi',   -np.pi, np.pi, valinit=pdict2['phase'], valfmt='%1.3f')
    sfcoh  = Slider(axfcoh, 'fcoh',      0.,  0.05, valinit=pdict2['fcoh'],  valfmt='%1.3f')
    swidth = Slider(axwdth, 'width',     0.,  3,    valinit=pdict2['width']*10**3,  valfmt='%1.3f')

    def update1(val):
        pdict1.update({'b' : [1, sb1.val, sb2.val], 'fbkg' : sfbkg.val, 'mass': smass.val})
        _, _, total = model(x0, **pdict1)
        tot1.set_ydata(total)
        fig.canvas.draw_idle()

    def update2(val):
        pdict2.update({'phase': sphi.val, 'fcoh' : sfcoh.val, 'width': swidth.val*10**-3})
        _, _, total = model(x0, **pdict2)
        tot2.set_ydata(total)
        fig.canvas.draw_idle()

    sphi.on_changed(update2)
    sfcoh.on_changed(update2)
    swidth.on_changed(update2)
    
    sb1.on_changed(update1)
    sb2.on_changed(update1)
    sfbkg.on_changed(update1)
    smass.on_changed(update1)

    plt.show()

def main():
    """ """
    show_model()

if __name__ == '__main__':
    main()
