

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
import SynthObs.Morph 
from SynthObs.Morph import measure 


cosmo = FLARE.default_cosmo()




model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model


filters = ['JWST.NIRCAM.F150W']

z = 8.

F = FLARE.filters.add_filters(filters, new_lam = model.lam * (1.+z)) 

PSF = SynthObs.Morph.webbPSFs(F['filters'], 101) 


model.create_Fnu_grid(F, z, cosmo)

test = SynthObs.test_data() # --- read in some test data

Fnu = {f: models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters}


img = SynthObs.Morph.observed_image(test.X, test.Y, Fnu['JWST.NIRCAM.F150W'], 'JWST.NIRCAM.F150W', cosmo, redshift = 8., Ndim = 101, smoothed = True, show = True, PSFs = PSF)


plt.imshow(img.img['JWST.NIRCAM.F150W'])
plt.savefig('webbnopsf.png')
plt.clf()


plt.imshow(img.psf_img['JWST.NIRCAM.F150W'])
plt.savefig('webbpsf.png')
plt.clf()

# 
# 
# img = SynthObs.Morph.physical_image(test.X, test.Y, test.Masses, smoothed = True, show = True)
# 
# m = measure.intrinsic(img)
# 
# m.detect_sources()
# 
# print(m.r_e()) # --- measure effective_radius in several different ways
# 
# 


