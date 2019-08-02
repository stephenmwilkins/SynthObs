

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import FLARE
import FLARE.filters
import SynthObs
from SynthObs.SED import models

import SynthObs.Morph.images
import SynthObs.Morph.PSF 


z = 8.
h = 0.697
cosmo = FLARE.default_cosmo()
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model



f = 'JWST.NIRCAM.F150W'
f = 'HST.WFC3.f160w'
F = FLARE.filters.add_filters([f], new_lam = model.lam * (1.+z)) 
model.create_Fnu_grid(F, z, cosmo)
PSF = SynthObs.Morph.PSF.PSF(f) # creates a dictionary of instances of the webbPSF class

test = SynthObs.test_data() # --- read in some test data
L = models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) 


do_test1 = True

if do_test1:


    width_arcsec = 2. # "
    # pixel_scale = 0.06 # "
    # smoothing = ('convolved_gaussian', (1.5/h)/(1.+z))
    smoothing = ('adaptive', 8.)
    
    imgs = SynthObs.Morph.images.observed(f, cosmo, z, width_arcsec, smoothing = smoothing, verbose = True, PSF = PSF).particle(test.X, test.Y, L)



    np = 6
    fig, axes = plt.subplots(1, np, figsize = (4*np,4))
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

    axes[0].imshow(imgs.super.hist, interpolation = 'nearest')
    axes[1].imshow(imgs.super.simple, interpolation = 'nearest')
    axes[2].imshow(imgs.super.no_PSF, interpolation = 'nearest')
    axes[3].imshow(imgs.super.data, interpolation = 'nearest')
    axes[4].imshow(imgs.img.no_PSF, interpolation = 'nearest')
    axes[5].imshow(imgs.img.data, interpolation = 'nearest')

    for ax in axes:    
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    plt.show()
    








