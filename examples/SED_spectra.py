

import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import SynthObs
from SynthObs.SED import models

import FLARE
import FLARE.filters

import matplotlib.pyplot as plt






# --- initialise SED grid ---
#  this can take a long time do don't do it for every object

model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -
model.dust = {'slope': -1.0} # define dust curve

fesc = 0.0

# --- read in test data

test = SynthObs.test_data() # --- read in some test data

# --- calculate V-band (550nm) optical depth for each star particle
A = 5.2
test.tauVs = (10**A) * test.MetSurfaceDensities

# --- now generate the various SEDs (nebular, stellar, total) [THIS IS IN THE REST-FRAME]

o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, test.tauVs, fesc = 0.0)

# --- now calculate some broad band photometry [THIS IS IN THE REST-FRAME]

filters = ['FAKE.TH.'+f for f in ['FUV','NUV','V']] # --- define the filters. FAKE.FAKE are just top-hat filters using for extracting rest-frame quantities.

F = FLARE.filters.add_filters(filters, new_lam = model.lam)




# --------------------------------------------------------------
# -- now make a plot of the SED and the broad-band photometry in terms of rest-frame luminosity

plt.plot(np.log10(o.lam), np.log10(o.intrinsic.lnu), label = 'intrinsic', c='k', alpha = 0.2, lw=2, zorder = 0)
plt.plot(np.log10(o.lam), np.log10(o.no_ISM.lnu), label = 'no_ISM', c='r', alpha=0.5, lw=1, zorder = 1)
plt.plot(np.log10(o.lam), np.log10(o.no_BC.lnu), label = 'no_BC', c='b', alpha=0.5, lw=1, zorder = 1)
plt.plot(np.log10(o.lam), np.log10(o.total.lnu), label = 'total', c='0.5', lw=1, zorder = 2)



for sed in [o.intrinsic, o.no_ISM, o.no_BC, o.total]:

    print(f'--- {sed.description}')

    sed.get_Lnu(F) # generates Lnu (broad band luminosities)

    for f in filters:
        print(f'{f} {np.log10(sed.Lnu[f]):.2f}')
        if sed == o.total:
            plt.scatter(np.log10(F[f].pivwv()), np.log10(sed.Lnu[f]), edgecolor = 'k', zorder = 3, label = f)




plt.xlim([2., 4.3])
plt.ylim([np.max(np.log10(o.total.lnu))-5., np.max(np.log10(o.total.lnu))+0.3])

plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
plt.ylabel(r'$\log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1})$')

plt.legend()
# plt.savefig('figs/SED_spectra_lnu.pdf')
plt.show()



# --------------------------------------------------------------
# -- now make a plot of the observed SED and the broad-band photometry

print()
print()

cosmo = FLARE.default_cosmo()

z = 8.

o.total.get_fnu(cosmo, z) # generates lamz and fnu

plt.plot(np.log10(o.total.lamz), o.total.fnu, label = 'total', lw=1, zorder = 1)


filters = FLARE.filters.NIRCam_W

F = FLARE.filters.add_filters(filters, new_lam = o.model.lam * (1. + z)) # --- NOTE: need to give it the redshifted

o.total.get_Fnu(F) # generates Fnu (broad band fluxes)

for f in filters:
    print(f'{f} {o.total.Fnu[f]:.2f}')
    plt.scatter(np.log10(F[f].pivwv()), o.total.Fnu[f], edgecolor = 'k', zorder = 2, label = f)

plt.xlim([3.6, 4.5])
plt.ylim([0, max(o.total.Fnu.values())*2.])

plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
plt.ylabel(r'$\log_{10}(f_{\nu}/nJy)$')

plt.legend()
# plt.savefig('figs/SED_spectra_fnu.pdf')
plt.show()
