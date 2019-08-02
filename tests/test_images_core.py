

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import SynthObs
import SynthObs.Morph.images


test = SynthObs.test_data() # --- read in some test data



do_test1 = True

if do_test1:

    redshift = 8.
    h = 0.697
    resolution = 0.1 # kpc
    ndim = 50 # pixels
    
    for smoothing in [('convolved_gaussian', (1.5/h)/(1.+redshift)), ('adaptive', 8.)]:
    
        img = SynthObs.Morph.images.core(test.X, test.Y, test.Masses, resolution = resolution, ndim = ndim, smoothing = smoothing, verbose = True)

        fig, axes = plt.subplots(1, 3, figsize = (6,2))
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    
        axes[0].imshow(img.hist, interpolation = 'nearest')
        axes[1].imshow(img.simple, interpolation = 'nearest')
        axes[2].imshow(img.data, interpolation = 'nearest')
    
        for ax in axes:    
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    
    
        plt.show()
    





# model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
# model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model
# 







