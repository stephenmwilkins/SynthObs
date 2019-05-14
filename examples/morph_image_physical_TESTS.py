

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import SynthObs
import SynthObs.Morph.images  



test = SynthObs.test_data() # --- read in some test data

redshift = 8.
h = 0.697



size = 5
resolution = 0.05
Ndim = int(size // resolution)

smoothing = 'gaussian'
smoothing_parameter = (1.5/h)/(1.+redshift) # gaussian FWHM




# ------ Make stellar mass vs. recent SF comparison




img_fixed = SynthObs.Morph.images.physical_individual(test.X, test.Y, test.Masses, resolution = resolution, Ndim = Ndim, smoothing = smoothing, smoothing_parameter = smoothing_parameter)

plt.imshow(img_fixed.data)
plt.show()


smoothing = 'adaptive'
smoothing_parameter = 4 # nearest neighbour 

img_adaptive = SynthObs.Morph.images.physical_individual(test.X, test.Y, test.Masses, resolution = resolution, Ndim = Ndim, smoothing = smoothing, smoothing_parameter = smoothing_parameter)

plt.imshow(img_fixed.data)
plt.show()



R = np.log10(img_adaptive.data/img_fixed.data)

plt.imshow(R, vmin=-0.5, vmax=0.5)
plt.show()



