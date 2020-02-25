

import numpy as np

import sys
import os


import SynthObs
from SynthObs.SED import models

import FLARE





# --- initialise model with SPS model and IMF. Set verbose = True to see a list of available lines.

# m = models.EmissionLines('BPASSv2.2.1.binary/ModSalpeter_300', verbose = False)
m = models.EmissionLines('BPASSv2.2.1.binary/ModSalpeter_300', verbose = False)


# --- read in test data based on BLUETIDES

test = SynthObs.test_data()

# UNITS:
# masses = star particle mass in M_sol
# ages = star particle age in Myr
# Z = star mass fraction in metals
# tauV = V-band (550nm) optical depth for each star particle

# --- calculate intrinsic quantities

o = m.get_line_luminosity('HI6563', test.Masses, test.Ages, test.Metallicities, verbose = True) # intrinsic line luminosities

# --- can also specify a doublet

o = m.get_line_luminosity('OII3726,OII3729', test.Masses, test.Ages, test.Metallicities, verbose = True) # intrinsic line luminosities


# --- calculate dust attenuated quantities

print('-'*30)
print('Calculate attenuated lines')

dust_BC = ('simple', {'slope': -1.0})
dust_ISM = ('simple', {'slope': -1.0})


m = models.EmissionLines('BPASSv2.2.1.binary/ModSalpeter_300', dust_BC = dust_BC, dust_ISM = dust_ISM, verbose = False)

# --- for BLUETIDES we found this gives a good fit to the LF at z=8
test.tauVs_ISM = (10**5.2) * test.MetSurfaceDensities
test.tauVs_BC = 2.0 * (test.Metallicities/0.01)

o = m.get_line_luminosity('HI6563', test.Masses, test.Ages, test.Metallicities, tauVs_BC = test.tauVs_BC, tauVs_ISM = test.tauVs_ISM, verbose = True) # intrinsic line luminosities




# --- Multiple lines at convenience

# o = m.get_line_luminosities(['HI6563','OII3726,OII3729'], test.Masses, test.Ages, test.Metallicities, tauVs_BC = test.tauVs_BC, tauVs_ISM = test.tauVs_ISM, verbose = True)

# print(o)
