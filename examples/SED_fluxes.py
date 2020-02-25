

import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import SynthObs
from SynthObs.SED import models

import FLARE
import FLARE.filters

import matplotlib.pyplot as plt




model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -
model.dust_BC = ('simple', {'slope': -1.0}) # define dust curve
model.dust_ISM = ('simple', {'slope': -1.0}) # define dust curve




# --- read in test data

test = SynthObs.test_data() # --- read in some test data


# --- create observed frame fluxes

z = 8.

F = FLARE.filters.add_filters(FLARE.filters.NIRCam_W, new_lam = model.lam * (1. + z))

cosmo = FLARE.default_cosmo()

model.create_Fnu_grid(F, z, cosmo) # --- create new Fnu grid for each filter. In units of nJy/M_sol

test.tauVs_ISM = (10**5.2) * test.MetSurfaceDensities
test.tauVs_BC = 2.0 * (test.Metallicities/0.01)

Fnu = models.generate_Fnu(model, test.Masses, test.Ages, test.Metallicities, test.tauVs_ISM, test.tauVs_BC, F, fesc = 0.0) # --- calculate rest-frame flux of each object in nJy

for f in F['filters']:
    print(f'{f} {Fnu[f]:.2f}')


# nJy
# This code:
# JWST.NIRCAM.F070W 0.02
# JWST.NIRCAM.F090W 0.24
# JWST.NIRCAM.F115W 49.12
# JWST.NIRCAM.F150W 75.72
# JWST.NIRCAM.F200W 79.48
# JWST.NIRCAM.F277W 90.09
# JWST.NIRCAM.F356W 133.18
# JWST.NIRCAM.F444W 183.70


# SED_spectra:
# JWST.NIRCAM.F070W 0.04
# JWST.NIRCAM.F090W 0.25
# JWST.NIRCAM.F115W 50.22
# JWST.NIRCAM.F150W 75.71
# JWST.NIRCAM.F200W 79.46
# JWST.NIRCAM.F277W 90.12
# JWST.NIRCAM.F356W 133.61
# JWST.NIRCAM.F444W 202.32
