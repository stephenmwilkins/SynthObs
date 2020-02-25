

import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import SynthObs
from SynthObs.SED import models

import FLARE
import FLARE.filters
from FLARE.SED import dust_curves

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm


# --- read in test data
test = SynthObs.test_data() # --- read in some test data
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Original case where the only dust is in the ISM

this = False
if this:

    model.dust_ISM = ('simple', {'slope': -1.0}) # define dust curve

    # --- calculate V-band (550nm) optical depth for each star particle
    A = 5.2
    test.tauVs_ISM = (10**A) * test.MetSurfaceDensities

    # --- general model spectra
    o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, tauVs_ISM = test.tauVs_ISM, fesc = 0.0)


    # --- plot escape fraction


    plt.plot(np.log10(o.lam), np.log10(o.f_esc), lw=2, c='k')

    plt.axhline(0.0,c='k',alpha=0.1)
    plt.xlim([3., 4.3])
    plt.ylim([-2.0, 0.1])

    plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\log_{10}(L_{\nu}^{intrinsic}/L_{\nu}^{attenuated})$')

    plt.show()


    # --- plot attenuation curve

    plt.plot(np.log10(o.lam), o.tau_relV, label = 'attenuation', lw=2, c='k')

    # --- add extinction curves

    lam = np.arange(1000.,10000.,1)

    for curve in dust_curves.curves:

        plt.plot(np.log10(lam), getattr(dust_curves, curve)().tau(lam), label = curve, lw=1, alpha=0.5)

    plt.axhline(1.0,c='k',alpha=0.1)
    plt.axvline(np.log10(5500),c='k',alpha=0.1)
    plt.xlim([3., 4.3])
    plt.ylim([-0.1, 5.])

    plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\tau/\tau_V$')

    plt.legend()
    plt.show()





# ----------------------------------------------------------------------------------------------------------------------------------------------
# NOW TRY VARIOUS EXTINCTION CURVES, STILL JUST INCLUDING DUST IN THE ISM

this = False
if this:

    for dust in [('simple', {'slope': -1.3}), ('simple', {'slope': -1.0}), ('simple', {'slope': -0.7}), ('SMC_Pei92', {}), ('MW_N18', {})]:

        dust_model, dust_model_params = dust
        model.dust_ISM = dust # update dust model

        o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, tauVs_ISM = test.tauVs_ISM, fesc = 0.0)

        plt.plot(np.log10(o.lam), o.tau_relV, label = f'{dust_model} {dust_model_params}')


    # --- add intrinsic dust curves for comparison

    lam = np.arange(1000.,10000.,1)

    for curve in dust_curves.curves:
        plt.plot(np.log10(lam), getattr(dust_curves, curve)().tau(lam), label = curve, lw=1, alpha=0.5)

    plt.axhline(1.0,c='k',alpha=0.1)
    plt.axvline(np.log10(1500),c='k',alpha=0.1)
    plt.axvline(np.log10(2500),c='k',alpha=0.1)
    plt.axvline(np.log10(5500),c='k',alpha=0.1)
    plt.xlim([3., 4.3])
    plt.ylim([-0.1, 5.])

    plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\tau/\tau_V$')

    plt.legend()
    plt.show()




# ----------------------------------------------------------------------------------------------------------------------------------------------
# NOW ADD A BC COMPONENT AND COMPARE THE TWO MODELS

this = False
if this:

    # --- set both BC and ISM to have the same extinction curve

    model.dust_ISM = ('simple', {'slope':-1})
    model.dust_BC = ('simple', {'slope':-1})

    # --- basic, no extra BC

    A = 5.2
    test.tauVs_ISM = (10**A) * test.MetSurfaceDensities
    o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, tauVs_ISM = test.tauVs_ISM, fesc = 0.0)

    plt.plot(np.log10(o.lam), o.tau_relV, label = f'basic (slope=-1)')

    # --- BC optical depth
    BC = 2.0
    test.tauVs_BC = BC * (test.Metallicities/0.01) # only gets applied to young particles

    o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, tauVs_ISM = test.tauVs_ISM, tauVs_BC = test.tauVs_BC, fesc = 0.0)

    plt.plot(np.log10(o.lam), o.tau_relV, label = fr'+birth cloud (slope=-1)')


    # --- set both BC and ISM to have different extinction curves

    model.dust_ISM = ('simple', {'slope':-0.7})
    model.dust_BC = ('simple', {'slope':-1.3})

    # --- basic, no extra BC

    A = 5.2
    test.tauVs_ISM = (10**A) * test.MetSurfaceDensities
    o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, tauVs_ISM = test.tauVs_ISM, fesc = 0.0)

    plt.plot(np.log10(o.lam), o.tau_relV, label = f'basic (slope=-0.7)')

    # --- BC optical depth
    BC = 2.0
    test.tauVs_BC = BC * (test.Metallicities/0.01) # only gets applied to young particles

    o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, tauVs_ISM = test.tauVs_ISM, tauVs_BC = test.tauVs_BC, fesc = 0.0)

    plt.plot(np.log10(o.lam), o.tau_relV, label = fr'+birth cloud (slope=-1.3)')


    plt.axhline(1.0,c='k',alpha=0.1)
    plt.axvline(np.log10(1500),c='k',alpha=0.1)
    plt.axvline(np.log10(2500),c='k',alpha=0.1)
    plt.axvline(np.log10(5500),c='k',alpha=0.1)
    plt.xlim([3., 4.3])
    plt.ylim([-0.1, 5.])

    plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\tau/\tau_{V}$')

    plt.legend()
    plt.show()





# --- EXPLORE DIFFERENT T_BC

this = True
if this:

    # --- set both BC and ISM to have the same extinction curve

    model.dust_ISM = ('simple', {'slope':-1})
    model.dust_BC = ('simple', {'slope':-1})

    # --- basic, no extra BC

    A = 5.2
    test.tauVs_ISM = (10**A) * test.MetSurfaceDensities


    # --- BC optical depth
    BC = 2.0
    test.tauVs_BC = BC * (test.Metallicities/0.01) # only gets applied to young particles

    for log10t_BC in [5.,6.,7.,8.,9.]:
        o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, tauVs_ISM = test.tauVs_ISM, tauVs_BC = test.tauVs_BC, fesc = 0.0, log10t_BC = log10t_BC)
        plt.plot(np.log10(o.lam), o.tau_relV, label = fr'{log10t_BC}')

    plt.axhline(1.0,c='k',alpha=0.1)
    plt.axvline(np.log10(1500),c='k',alpha=0.1)
    plt.axvline(np.log10(2500),c='k',alpha=0.1)
    plt.axvline(np.log10(5500),c='k',alpha=0.1)
    plt.xlim([3., 4.3])
    plt.ylim([-0.1, 5.])

    plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\tau/\tau_{V}$')

    plt.legend()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------
# Explore a range of BC values

this = False
if this:

    model.dust_ISM = ('simple', {'slope':-1})
    model.dust_BC = ('simple', {'slope':-1})

    # --- basic, no extra BC

    A = 5.2
    test.tauVs_ISM = (10**A) * test.MetSurfaceDensities
    o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, tauVs_ISM = test.tauVs_ISM, fesc = 0.0)

    plt.plot(np.log10(o.lam), o.tau_relV, label = f'basic')

    # --- now add BC with various strengths

    norm = mpl.colors.Normalize(vmin=0.0, vmax=5.0)

    for BC in np.arange(0.0, 5.0, 0.5):

        # --- ISM optical depth
        A = 5.2
        test.tauVs_ISM = (10**A) * test.MetSurfaceDensities

        # --- BC optical depth
        test.tauVs_BC = BC * (test.Metallicities/0.01) # only gets applied to young particles

        o = models.generate_SED(model, test.Masses, test.Ages, test.Metallicities, tauVs_ISM = test.tauVs_ISM, tauVs_BC = test.tauVs_BC, fesc = 0.0)

        plt.plot(np.log10(o.lam), o.tau_relV, label = fr'extra birth cloud $\tau={BC}$', c=cm.coolwarm(norm(BC)))


    plt.axhline(1.0,c='k',alpha=0.1)
    plt.axvline(np.log10(1500),c='k',alpha=0.1)
    plt.axvline(np.log10(2500),c='k',alpha=0.1)
    plt.axvline(np.log10(5500),c='k',alpha=0.1)
    plt.xlim([3., 4.3])
    plt.ylim([-0.1, 5.])

    plt.xlabel(r'$\log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\tau/\tau_{1500}$')

    plt.legend()
    plt.show()
