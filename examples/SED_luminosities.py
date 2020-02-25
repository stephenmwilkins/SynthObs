

import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import SynthObs
from SynthObs.SED import models

import FLARE
import FLARE.filters

import matplotlib.pyplot as plt


"This is an more efficient way of calculating the broadband luminosities of galaxies. It produces the same answer as generate_SED within ~0.05 dex"


model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300', path_to_SPS_grid = FLARE.FLARE_dir + '/data/SPS/nebular/3.0/') # DEFINE SED GRID -
# model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300', path_to_SPS_grid = FLARE.FLARE_dir + '/data/SPS/nebular/2.0/Z_refQ_wdust/') # DEFINE SED GRID -
model.dust_ISM = ('simple', {'slope': -1.0})
model.dust_BC = ('simple', {'slope': -1.0})

# --- read in test data

test = SynthObs.test_data() # --- read in some test data


# --- create rest-frame luminosities

filters = ['FAKE.TH.'+f for f in ['FUV','NUV','V']] # --- define the filters. FAKE.FAKE are just top-hat filters using for extracting rest-frame quantities.

F = FLARE.filters.add_filters(filters, new_lam = model.lam)

model.create_Lnu_grid(F) # --- create new L grid for each filter. In units of erg/s/Hz



print('--- Pure stellar luminosities')

# --- Pure stellar
test.tauVs_ISM = np.zeros(test.Masses.shape)
test.tauVs_BC = np.zeros(test.Masses.shape)
Lnu = models.generate_Lnu(model, test.Masses, test.Ages, test.Metallicities, test.tauVs_ISM, test.tauVs_BC, F, fesc = 1.0) # --- calculate rest-frame Luminosity. In units of erg/s/Hz
for f in F['filters']:
    print(f'{f} {np.log10(Lnu[f]):.2f}')

print('--- With just HII region')

# --- Intrinsic (just stellar + HII)
test.tauVs_ISM = np.zeros(test.Masses.shape)
test.tauVs_BC = np.zeros(test.Masses.shape)
Lnu = models.generate_Lnu(model, test.Masses, test.Ages, test.Metallicities,  test.tauVs_ISM, test.tauVs_BC, F,  fesc = 0.0) # --- calculate rest-frame Luminosity. In units of erg/s/Hz
for f in F['filters']:
    print(f'{f} {np.log10(Lnu[f]):.2f}')


print('--- With just BC dust')

# --- Intrinsic (just stellar + HII)
test.tauVs_ISM = np.zeros(test.Masses.shape)
test.tauVs_BC = 2.0 * (test.Metallicities/0.01)
Lnu = models.generate_Lnu(model, test.Masses, test.Ages, test.Metallicities,  test.tauVs_ISM, test.tauVs_BC, F,  fesc = 0.0) # --- calculate rest-frame Luminosity. In units of erg/s/Hz
for f in F['filters']:
    print(f'{f} {np.log10(Lnu[f]):.2f}')

print('--- Total')

# --- TOTAL
test.tauVs_ISM = (10**5.2) * test.MetSurfaceDensities
test.tauVs_BC = 2.0 * (test.Metallicities/0.01)
Lnu = models.generate_Lnu(model, test.Masses, test.Ages, test.Metallicities,  test.tauVs_ISM, test.tauVs_BC, F, fesc = 0.0) # --- calculate rest-frame Luminosity. In units of erg/s/Hz

for f in F['filters']:
    print(f'{f} {np.log10(Lnu[f]):.2f}')



# REPORTED BY THIS CODE
# --- Pure stellar luminosities
# FAKE.TH.FUV 29.56
# FAKE.TH.NUV 29.42
# FAKE.TH.V 29.38
# --- With just HII region
# FAKE.TH.FUV 29.61
# FAKE.TH.NUV 29.50
# FAKE.TH.V 29.44
# --- With just BC dust
# FAKE.TH.FUV 29.28
# FAKE.TH.NUV 29.25
# FAKE.TH.V 29.35
# --- Total
# FAKE.TH.FUV 28.82
# FAKE.TH.NUV 28.87
# FAKE.TH.V 29.10


# REPORTED BY SED_SPECTRA
# --- intrinsic (stellar) SED
# FAKE.TH.FUV 29.56
# FAKE.TH.NUV 29.42
# FAKE.TH.V 29.38
# --- SED with nebular emission (HII) but no BC or ISM dust
# FAKE.TH.FUV 29.61
# FAKE.TH.NUV 29.50
# FAKE.TH.V 29.44
# --- SED with nebular emission (HII), BC dust, but no ISM dust
# FAKE.TH.FUV 29.28
# FAKE.TH.NUV 29.25
# FAKE.TH.V 29.35
# --- SED with no nebular emission (HII) or dusty BC (same as setting fesc=1.0)
# FAKE.TH.FUV 29.01
# FAKE.TH.NUV 28.97
# FAKE.TH.V 29.11
# --- SED including both birth cloud and ISM dust
# FAKE.TH.FUV 28.82
# FAKE.TH.NUV 28.87
# FAKE.TH.V 29.10
