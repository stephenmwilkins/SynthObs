

import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import synthobs
from synthobs.sed import models

import flare
import flare.filters


import matplotlib.pyplot as plt


# --- initialise SED grid ---
#  this can take a long time do don't do it for every object
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -

# --- read in test data
test = synthobs.test_data() # --- read in some test data

# --- calculate the ionising photon luminosity
log10Q = models.generate_log10Q(model, test.Masses, test.Ages, test.Metallicities)


print(f'log10(M*): {np.log10(np.sum(test.Masses))}')
print(f'log10Q (direct): {log10Q}')

# --- calculate the full SEDs


o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, fesc = 1.0)

log10Q_SED = o.stellar.return_log10Q()
print(f'log10Q (SED): {log10Q_SED}')




# --- calculate the UV luminosity

filters = ['FAKE.TH.FUV'] # --- define the filters. FAKE.FAKE are just top-hat filters using for extracting rest-frame quantities.

F = flare.filters.add_filters(filters, new_lam = model.lam)

model.create_Lnu_grid(F) # --- create new L grid for each filter. In units of erg/s/Hz

# --- Pure stellar

Lnu = models.generate_Lnu(model, F, test.Masses, test.Ages, test.Metallicities, fesc = 1.0)

log10LFUV = np.log10(Lnu['FAKE.TH.FUV']) # erg/s/Hz

print(f'log10LFUV: {log10LFUV}')
print(f'ionising photon production efficiecy: {log10Q-log10LFUV}') # should be ~25!
