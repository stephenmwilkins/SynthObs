

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import SynthObs.Morph 
import SynthObs.Morph.PSF as PSF
import FLARE.filters
from matplotlib.colors import LogNorm

width = 10

filters = FLARE.filters.NIRCam_W # all NIRCam wide filters

filters = ['JWST.NIRCAM.F115W','JWST.NIRCAM.F150W']

print(filters)

PSFs = PSF.Webb(filters, resampling_factor = 5) # creates a dictionary of instances of the webbPSF class


fig, axes = plt.subplots(1, len(filters), figsize = (len(filters)*2., 2))

fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

for ax, f in zip(axes.flatten(), filters):
    
    ax.imshow(PSFs[f].data,norm=LogNorm())
    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.text(0.5, 0.5, f.split('.')[-1], fontsize = 30, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


plt.show()


