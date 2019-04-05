

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
import SynthObs.Morph.PSF as PSF



cosmo = FLARE.default_cosmo()

width = 4. # size of cutout in "
z = 8.


model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model


filters = ['Euclid.NISP.H', 'HST.WFC3.f160w', 'JWST.NIRCAM.F150W']

F = FLARE.filters.add_filters(filters, new_lam = model.lam * (1.+z)) 

model.create_Fnu_grid(F, z, cosmo)




test = SynthObs.test_data() # --- read in some test data

Fnu = {f: models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters} # arrays of star particle fluxes in nJy


fig, axes = plt.subplots(1, len(filters), figsize = (len(filters)*2., 2))

fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)


xoffset = np.random.random() - 0.5
yoffset = np.random.random() - 0.5


resampling_factor = 1

for ax, f in zip(axes.flatten(), filters):

    observatory = f.split('.')[0]
    if observatory == 'JWST': observatory = 'Webb'
    if observatory == 'HST': observatory = 'Hubble'

    psf = getattr(PSF, observatory+'PSF')(f)
    
    
    img = SynthObs.Morph.images.observed_individual(test.X, test.Y, Fnu[f], f, cosmo, redshift = 8., width = width, resampling_factor = resampling_factor, PSF = psf, xoffset = xoffset, yoffset = yoffset)

    ax.imshow(img.data, interpolation = 'nearest')

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


fig.savefig('f/morph_image_comparison_resampling{0}.png'.format(resampling_factor), dpi = img.data.shape[0]*2)
plt.show()
fig.clf()
