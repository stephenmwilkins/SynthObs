

import numpy as np

import sys
import os


import SynthObs
from SynthObs.SED import models

import FLARE





# --- initialise model with SPS model and IMF. Set verbose = True to see a list of available lines.

# m = models.EmissionLines('BPASSv2.2.1.binary/ModSalpeter_300', verbose = False)
m = models.EmissionLines('BPASSv2.2.1.binary/ModSalpeter_300', verbose = False, path_to_SPS_grid = f'{FLARE.FLARE_dir}/data/SPS/nebular/2.0/Z_refQ_wdust/')


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

m.dust = {'slope': -1.0} # specify ISM dust model simple power-law dust curve

# --- for BLUETIDES we found this gives a good fit to the LF at z=8
A = 5.2
test.tauVs = (10**A) * test.MetSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle


o = m.get_line_luminosity('HI6563', test.Masses, test.Ages, test.Metallicities, tauVs = test.tauVs, verbose = True) # intrinsic line luminosities




# --- Multiple lines at convenience

o = m.get_line_luminosities(['HI6563','OII3726,OII3729'], test.Masses, test.Ages, test.Metallicities, tauVs = test.tauVs, verbose = True)

# print(o)
