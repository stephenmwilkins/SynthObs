

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


filters = ['FAKE.FAKE.'+f for f in ['1500']] # --- define the filters. FAKE.FAKE are just top-hat filters using for extracting rest-frame quantities.

F = FLARE.filters.add_filters(filters, new_lam = model.lam) 


model.create_Lnu_grid(F)


test = SynthObs.test_data() # --- read in some test data

L = {f: models.generate_Lnu_array(model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters}


# ------ Make mass image

img = SynthObs.Morph.physical_image(test.X, test.Y, test.Masses, smoothed = True, show = True)

m = measure.intrinsic(img)

m.detect_sources()

print(m.r_e()) # --- measure effective_radius in several different ways


# ------ Make rest-frame luminosity image

img = SynthObs.Morph.physical_image(test.X, test.Y, L['FAKE.FAKE.1500'], smoothed = True, show = True)

m = measure.intrinsic(img)

m.detect_sources()

print(m.r_e()) # --- measure effective_radius in several different ways





