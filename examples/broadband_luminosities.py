

import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SynthObs.SED import models
from FLARE.SED import filter_utility

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


# --- read in some test data
# --- these are 1D arrays containing values for a single object


MetSurfaceDensities = np.load(path_to_example_data+'/MetSurfaceDensities.npy') # surface density of metals along line of sight to each star particle. Used for calculating dust.
Ages = np.load(path_to_example_data+'/Ages.npy') # age of each star particle in Myr 
Metallicities = np.load(path_to_example_data+'/Metallicities.npy') # mass fraction of stars in metals (Z)
Masses = np.ones(Ages.shape) * 1E10*5.90556119E-05/0.697 # mass of each star particle in M_sol . NOTE: this value is for BLUETIDES in h-less units.

L = models.generate_L(model, Masses, Ages, Metallicities, MetSurfaceDensities, F)

print(L)



# --- now test speed

N = 100

start = time.time()
for i in range(N):
    L = models.generate_L(model, Masses, Ages, Metallicities, MetSurfaceDensities, F)
print((time.time() - start)/N)

# 
# # --- this is ~30 times slower!
#     
# start = time.time()
# for i in range(N):
#     o = models.generate_SED(model, Masses, Ages, Metallicities, MetSurfaceDensities)
#     o.total.get_Lnu(F) # generates Lnu (broad band luminosities)
# print((time.time() - start)/N)
















