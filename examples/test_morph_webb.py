

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


filters = FLARE.filters.NIRCam_W

print(filters)

width = 2. # size of cutout in "
z = 8.

F = FLARE.filters.add_filters(filters, new_lam = model.lam * (1.+z)) 

PSFs = SynthObs.Morph.webbPSFs(F['filters'], width) # creates a dictionary of instances of the webbPSF class



model.create_Fnu_grid(F, z, cosmo)

test = SynthObs.test_data() # --- read in some test data

Fnu = {f: models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters}


img = SynthObs.Morph.observed_images(test.X, test.Y, Fnu, filters, cosmo, redshift = 8., width = width, smoothed = True, show = False, PSFs = PSFs)





fig, axes = plt.subplots(1, len(filters), figsize = (len(filters)*2., 2))

fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

for ax, f in zip(axes.flatten(), filters):
    
    ax.imshow(img[f].psf_img)
    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.text(0.5, 0.85, f.split('.')[-1], fontsize = 15, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


fig.savefig('webb.png')
plt.show()



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


