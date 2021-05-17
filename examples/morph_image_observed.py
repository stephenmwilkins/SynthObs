

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flare
import flare.filters
import synthobs
from synthobs.sed import models

import synthobs.morph.images
import synthobs.morph.PSF



cosmo = flare.default_cosmo()
z = 8.


test = synthobs.test_data() # --- read in some test data

# --- calculate V-band (550nm) optical depth for each star particle
A = 5.2
test.tauVs = (10**A) * test.MetSurfaceDensities



model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -
model.dust = {'slope': -1.0} # define dust curve







do_test = False

if do_test:

    width_arcsec = 3.

    f = 'Webb.NIRCam.F150W'

    F = flare.filters.add_filters([f], new_lam = model.lam * (1.+z))

    PSF = synthobs.morph.PSF.PSF(f) # creates a dictionary of instances of the webbPSF class

    model.create_Fnu_grid(F, z, cosmo)

    tauVs_BC = np.zeros(len(test.tauVs))

    Fnu = models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.tauVs, tauVs_BC, F, f, fesc = 0.0)

    imgs = synthobs.morph.images.observed(f, cosmo, z, width_arcsec, smoothing = ('adaptive', 8.), verbose = True, PSF = PSF, super_sampling = 2).particle(test.X, test.Y, Fnu)


    fig, ax = plt.subplots(1, 1, figsize = (5,5))

    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


    ax.imshow(imgs.img.data, interpolation = 'nearest')
    plt.show()






do_comparison = True

if do_comparison:



    for filters, width_arcsec, name in zip([['Euclid.NISP.H', 'Hubble.WFC3.f160w', 'Webb.NIRCam.F150W'],['Spitzer.IRAC.ch1', 'Webb.NIRCam.F356W']], [5., 10.], ['H', '3.6']):

        F = flare.filters.add_filters(filters, new_lam = model.lam * (1.+z))

        PSFs = synthobs.morph.PSF.PSFs(F['filters']) # creates a dictionary of instances of the webbPSF class

        model.create_Fnu_grid(F, z, cosmo)

        tauVs_BC = np.zeros(len(test.tauVs))

        Fnu = {f: models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.tauVs, tauVs_BC, F, f, fesc = 0.0) for f in filters} # arrays of star particle fluxes in nJy

        imgs = synthobs.morph.images.particle(test.X, test.Y, Fnu, filters, cosmo, z, width_arcsec, smoothing = ('adaptive', 8.), verbose = True, PSFs = PSFs, super_sampling = 2)

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
#     filter_sets['Webb_NIRCam_s_W'] = flare.filters.NIRCam_s_W
#     filter_sets['Webb_NIRCam_l_W'] = flare.filters.NIRCam_l_W
#     filter_sets['Webb_MIRI'] = flare.filters.MIRI

    filter_sets['Hubble_WFC3NIR_W'] = flare.filters.WFC3NIR_W
#     filter_sets['Euclid_NISP'] = flare.filters.Euclid_NISP
#     filter_sets['Spitzer_IRAC'] = flare.filters.IRAC

    width_arcsec = 2. # size of cutout in "

    for filter_set in filter_sets.keys():

        print('-'*20)
        print(filter_set)

        filters = filter_sets[filter_set]
        print(filters)

        F = flare.filters.add_filters(filters, new_lam = model.lam * (1.+z))

        PSFs = synthobs.morph.PSF.PSFs(F['filters']) # creates a dictionary of instances of the webbPSF class

        model.create_Fnu_grid(F, z, cosmo)

        tauVs_BC = np.zeros(len(test.tauVs))
        Fnu = {f: models.generate_Fnu_array(model, test.Masses, test.Ages, test.Metallicities, test.tauVs, tauVs_BC, F, f) for f in filters} # arrays of star particle fluxes in nJy

        imgs = synthobs.morph.images.particle(test.X, test.Y, Fnu, filters, cosmo, z, width_arcsec, smoothing = ('adaptive', 8.), verbose = False, PSFs = PSFs, super_sampling = 2)

        imgs_dithered = synthobs.morph.images.particle(test.X, test.Y, Fnu, filters, cosmo, z, width_arcsec, resampling_factor=2, smoothing = ('adaptive', 8.), verbose = False, PSFs = PSFs, super_sampling = 2)


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
