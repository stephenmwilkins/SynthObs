

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import synthobs.morph
import flare.filters


width = 10

filters = flare.filters.NIRCam_W # all NIRCam wide filters

filters = ['JWST.NIRCAM.F115W','JWST.NIRCAM.F150W']

PSFs = synthobs.morph.webbPSFs(filters, width) # creates a dictionary of instances of the webbPSF class


fig, axes = plt.subplots(1, len(filters), figsize = (len(filters)*2., 2))

fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

for ax, f in zip(axes.flatten(), filters):

    ax.imshow(np.log10(PSFs[f].PSF))

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.text(0.5, 0.5, f.split('.')[-1], fontsize = 30, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


plt.show()
