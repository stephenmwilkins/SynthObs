

import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import SynthObs
from SynthObs.SED import models
from FLARE.SED import filter_utility
import SynthObs.Morph 
from SynthObs.Morph import measure 

import matplotlib.pyplot as plt

import time

data_dir = '/Users/stephenwilkins/Dropbox/Research/Data'
path_to_example_data = data_dir + '/package_example_data/SynthObs/'
path_to_SPS_grid = data_dir + '/SPS/nebular/1.0/Z/'
filter_path = data_dir + '/filters/'


model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300', path_to_SPS_grid = path_to_SPS_grid) # DEFINE SED GRID - 
model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model


filters = ['FAKE.FAKE.'+f for f in ['1500']] # --- define the filters. FAKE.FAKE are just top-hat filters using for extracting rest-frame quantities.

F = filter_utility.add_filters(filters, new_lam = model.lam, filter_path = filter_path) 


model.create_L(F)


test = SynthObs.test_data(path_to_example_data) # --- read in some test data

L = {f: models.generate_L_array(model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters}

print(L['FAKE.FAKE.1500'].shape)



img = SynthObs.Morph.physical_image(test.X, test.Y, test.Masses, smoothed = True, show = True)

m = measure.intrinsic(img)

m.detect_sources()

print(m.r_e()) # --- measure effective_radius in several different ways



