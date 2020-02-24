

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import FLARE
import FLARE.filters
import SynthObs
from SynthObs.SED import models

import SynthObs.Morph.images
import SynthObs.Morph.PSF



cosmo = FLARE.default_cosmo()
z = 8.


test = SynthObs.test_data() # --- read in some test data

# --- calculate V-band (550nm) optical depth for each star particle
A = 5.2
test.tauVs = (10**A) * test.MetSurfaceDensities



model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -
model.dust = {'slope': -1.0} # define dust curve







do_test = False

if do_test:

    width_arcsec = 3.

    f = 'JWST.NIRCAM.F150W'

    F = FLARE.filters.add_filters([f], new_lam = model.lam * (1.+z))

    PSF = SynthObs.Morph.PSF.PSF(f) # creates a dictionary of instances of the webbPSF class

    model.create_Fnu_grid(F, z, cosmo)

    Fnu = models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.tauVs, F, f, fesc = 0.0)

    imgs = SynthObs.Morph.images.observed(f, cosmo, z, width_arcsec, smoothing = ('adaptive', 8.), verbose = True, PSF = PSF, super_sampling = 2).particle(test.X, test.Y, Fnu)


    fig, ax = plt.subplots(1, 1, figsize = (5,5))

    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


    ax.imshow(imgs.img.data, interpolation = 'nearest')
    plt.show()






do_comparison = True

if do_comparison:



    for filters, width_arcsec, name in zip([['Euclid.NISP.H', 'HST.WFC3.f160w', 'JWST.NIRCAM.F150W'],['Spitzer.IRAC.ch1', 'JWST.NIRCAM.F356W']], [5., 10.], ['H', '3.6']):

        F = FLARE.filters.add_filters(filters, new_lam = model.lam * (1.+z))

        PSFs = SynthObs.Morph.PSF.PSFs(F['filters']) # creates a dictionary of instances of the webbPSF class

        model.create_Fnu_grid(F, z, cosmo)

        Fnu = {f: models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.tauVs, F, f, fesc = 0.0) for f in filters} # arrays of star particle fluxes in nJy

        imgs = SynthObs.Morph.images.particle(test.X, test.Y, Fnu, filters, cosmo, z, width_arcsec, smoothing = ('adaptive', 8.), verbose = True, PSFs = PSFs, super_sampling = 2)

        fig, axes = plt.subplots(1, len(filters), figsize = (len(filters)*2., 2))

        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

        for ax, f in zip(axes.flatten(), filters):

            ax.imshow(imgs[f].data, interpolation = 'nearest')

            print('{0}: {1:.2f}'.format(f, np.sum(imgs[f].data))) # total flux in nJy

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            # ax.text(0.5, 0.85, f.split('.')[-1], fontsize = 15, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        plt.savefig('figs/{0}.pdf'.format(name))
        plt.show()
        fig.clf()



do_filter_sets = True

if do_filter_sets:



    filter_sets = {}
#     filter_sets['Webb_NIRCam_s_W'] = FLARE.filters.NIRCam_s_W
#     filter_sets['Webb_NIRCam_l_W'] = FLARE.filters.NIRCam_l_W
#     filter_sets['Webb_MIRI'] = FLARE.filters.MIRI

    filter_sets['Hubble_WFC3NIR_W'] = FLARE.filters.WFC3NIR_W
#     filter_sets['Euclid_NISP'] = FLARE.filters.Euclid_NISP
#     filter_sets['Spitzer_IRAC'] = FLARE.filters.IRAC

    width_arcsec = 2. # size of cutout in "

    for filter_set in filter_sets.keys():

        print('-'*20)
        print(filter_set)

        filters = filter_sets[filter_set]
        print(filters)

        F = FLARE.filters.add_filters(filters, new_lam = model.lam * (1.+z))

        PSFs = SynthObs.Morph.PSF.PSFs(F['filters']) # creates a dictionary of instances of the webbPSF class

        model.create_Fnu_grid(F, z, cosmo)

        Fnu = {f: models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.tauVs, F, f) for f in filters} # arrays of star particle fluxes in nJy

        imgs = SynthObs.Morph.images.particle(test.X, test.Y, Fnu, filters, cosmo, z, width_arcsec, smoothing = ('adaptive', 8.), verbose = False, PSFs = PSFs, super_sampling = 2)

        imgs_dithered = SynthObs.Morph.images.particle(test.X, test.Y, Fnu, filters, cosmo, z, width_arcsec, resampling_factor=2, smoothing = ('adaptive', 8.), verbose = False, PSFs = PSFs, super_sampling = 2)


        fig, axes = plt.subplots(2, len(filters), figsize = (len(filters)*2., 4))

        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

        for i, f in enumerate(filters):

            axes[0, i].imshow(imgs[f].data, interpolation = 'nearest')
            axes[1, i].imshow(imgs_dithered[f].data, interpolation = 'nearest')

            print('{0}: {1:.2f}'.format(f, np.sum(imgs[f].data))) # total flux in nJy

            for j in range(2):
                axes[j,i].get_xaxis().set_ticks([])
                axes[j,i].get_yaxis().set_ticks([])
            axes[0,i].text(0.5, 0.85, f.split('.')[-1], fontsize = 15, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=axes[0,i].transAxes)

        plt.savefig('figs/{0}.pdf'.format(filter_set), dpi = imgs[f].data.shape[0]*2)
        plt.show()
        fig.clf()
