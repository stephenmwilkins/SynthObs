

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import SynthObs
import SynthObs.Morph 
from SynthObs.SED import models
from SynthObs.Morph import measure 







test = SynthObs.test_data() # --- read in some test data







# ------ Make rest-frame luminosity image

imgs = {}

imgs['mass'] = SynthObs.Morph.physical_image(test.X, test.Y, test.Masses, Ndim = 50)

s = test.Ages<10.
imgs['sfr'] = SynthObs.Morph.physical_image(test.X[s], test.Y[s], test.Masses[s], Ndim = 50)

N = len(imgs.keys())
fig, axes = plt.subplots(1, N, figsize = (N*2., 2))
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

for ax, f in zip(axes.flatten(), ['mass', 'sfr']):   
    ax.imshow(imgs[f].img)  
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.text(0.5, 0.9, f, fontsize = 30, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


fig.savefig('physical.png')
plt.show()



