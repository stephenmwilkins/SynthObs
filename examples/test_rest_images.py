

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import FLARE.filters
import SynthObs
import SynthObs.Morph 
from SynthObs.SED import models
from SynthObs.Morph import measure 








model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model


filters = FLARE.filters.FAKE

F = FLARE.filters.add_filters(filters, new_lam = model.lam) 


model.create_Lnu_grid(F)


test = SynthObs.test_data() # --- read in some test data

L = {f: models.generate_Lnu_array(model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters}


# ------ Make rest-frame luminosity image

IMGs = SynthObs.Morph.physical_images(test.X, test.Y, L, filters, Ndim = 50)

fig, axes = plt.subplots(1, len(filters), figsize = (len(filters)*2., 2))
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

for ax, f in zip(axes.flatten(), filters):   
    ax.imshow(IMGs[f].img)  
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.text(0.5, 0.9, f.split('.')[-1], fontsize = 30, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


fig.savefig('rest.png')
# plt.show()



