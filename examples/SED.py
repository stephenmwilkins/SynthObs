

import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SynthObs.SED import models
from FLARE.SED import filter_utility

import matplotlib.pyplot as plt


data_dir = '/Users/stephenwilkins/Dropbox/Research/Data'

path_to_example_data = data_dir + '/package_example_data/SynthObs/'
path_to_SPS_grid = data_dir + '/SPS/nebular/1.0/Z/'
filter_path = data_dir + '/filters/'






# --- read in some test data
# --- these are 1D arrays containing values for a single object


MetSurfaceDensities = np.load(path_to_example_data+'/MetSurfaceDensities.npy') # surface density of metals along line of sight to each star particle. Used for calculating dust.
Ages = np.load(path_to_example_data+'/Ages.npy') # age of each star particle in Myr 
Metallicities = np.load(path_to_example_data+'/Metallicities.npy') # mass fraction of stars in metals (Z)
Masses = np.ones(Ages.shape) * 1E10*5.90556119E-05/0.697 # mass of each star particle in M_sol . NOTE: this value is for BLUETIDES in h-less units.

# --- initialise SED grid ---
#  this can take a long time do don't do it for every object

model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300', path_to_SPS_grid = path_to_SPS_grid) # DEFINE SED GRID - 
model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model



# --- now generate the various SEDs (nebular, stellar, total) [THIS IS IN THE REST-FRAME]

o = models.generate_SED(model, Masses, Ages, Metallicities, MetSurfaceDensities)

# --- now calculate some broad band photometry [THIS IS IN THE REST-FRAME]

filters = ['FAKE.FAKE.'+f for f in ['1500','2500','V']] # --- define the filters. FAKE.FAKE are just top-hat filters using for extracting rest-frame quantities.

F = filter_utility.add_filters(filters, new_lam = model.lam, filter_path = filter_path) 

o.total.get_Lnu(F) # generates Lnu (broad band luminosities)


# --------------------------------------------------------------
# -- now make a plot of the SED and the broad-band photometry in terms of rest-frame luminosity

plt.plot(np.log10(o.lam), np.log10(o.intrinsic_total.lnu), label = 'intrinsic total', c='k', alpha = 0.2, lw=2, zorder = 0)
plt.plot(np.log10(o.lam), np.log10(o.total.lnu), label = 'total', lw=1, zorder = 1)
plt.plot(np.log10(o.lam), np.log10(o.stellar.lnu), label = 'stellar', lw=1, zorder = 1)

for f in filters: plt.scatter(np.log10(F[f].pivwv()), np.log10(o.total.Lnu[f]), edgecolor = 'k', zorder = 2, label = f)

plt.xlim([2., 4.3])
plt.ylim([np.max(np.log10(o.total.lnu))-5., np.max(np.log10(o.total.lnu))+0.3])

plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
plt.ylabel(r'$\log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1})$')

plt.legend()
plt.show() 



# --------------------------------------------------------------
# -- now make a plot of the SED and the broad-band photometry 

from astropy.cosmology import WMAP9 as cosmo

z = 8.

o.total.get_fnu(cosmo, z) # generates lamz and fnu

plt.plot(np.log10(o.total.lamz), o.total.fnu, label = 'total', lw=1, zorder = 1)


filters = ['HST.ACS.'+f for f in ['f850lp']] 
filters += ['HST.WFC3.'+f for f in ['f105w', 'f125w', 'f140w', 'f160w']]

F = filter_utility.add_filters(filters, new_lam = o.model.lam * (1. + z), filter_path = filter_path) # --- NOTE: need to give it the redshifted 

o.total.get_Fnu(F) # generates Fnu (broad band fluxes)

for f in filters: plt.scatter(np.log10(F[f].pivwv()), o.total.Fnu[f], edgecolor = 'k', zorder = 2, label = f)

plt.xlim([3.6, 4.5])
plt.ylim([0, max(o.total.Fnu.values())*2.])

plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
plt.ylabel(r'$\log_{10}(f_{\nu}/nJy)$')

plt.legend()
plt.show() 




