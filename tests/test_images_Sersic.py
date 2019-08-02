

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

from photutils import CircularAperture
from photutils import aperture_photometry

z = 8.
h = 0.697
cosmo = FLARE.default_cosmo()
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model



f = 'JWST.NIRCAM.F150W'
f = 'HST.WFC3.f160w'
F = FLARE.filters.add_filters([f], new_lam = model.lam * (1.+z)) 
PSF = SynthObs.Morph.PSF.PSF(f) # creates a dictionary of instances of the webbPSF class


L = 1.

do_test1 = True
show = True

if do_test1:

    width_arcsec = 2. # "
    pixel_scale = 0.06
    
    
    p = {'r_eff':  1.0, 'ellip': 0.6, 'theta': np.pi/4., 'n': 0.5}
    # p = {'r_eff':  1.0, 'ellip': 0.3, 'theta': 0.0, 'n': 1.}
    #p = {'r_eff':  1.0, 'ellip': 0.3, 'theta': np.pi/4., 'n': 1.5 }
    
    imgs = SynthObs.Morph.images.observed(f, cosmo, z, width_arcsec, pixel_scale = pixel_scale, verbose = True, PSF = PSF).Sersic(L, p)

    if show:
        npanels = 5
        fig, axes = plt.subplots(1, npanels, figsize = (4*npanels,4))
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

        axes[1].imshow(imgs.super.no_PSF, interpolation = 'nearest')
        axes[2].imshow(imgs.super.data, interpolation = 'nearest')
        axes[3].imshow(imgs.img.no_PSF, interpolation = 'nearest')
        axes[4].imshow(imgs.img.data, interpolation = 'nearest')

        for ax in axes:    
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])


        # plt.savefig('test_images2.pdf')
        plt.show()
    

    # --- measure size


    for img, label in zip([imgs.super, imgs.img], ['super resolution', 'observed']):

        print('-'*10, label)

        data = img.data

        ndim = data.shape[0]

        print('image flux: {0:.2f}'.format(np.sum(data)))
        print('pixel scale/kpc: {0:.2f}'.format(img.pixel_scale_kpc))
        print('ndim: {0}'.format(ndim))

        from photutils import data_properties

        cat = data_properties(data)

        print('a: {0:.2f}'.format(cat.semimajor_axis_sigma.value * img.pixel_scale_kpc)) # only equal to r_e_major for n=1 
        print('b: {0:.2f}'.format(cat.semiminor_axis_sigma.value * img.pixel_scale_kpc)) # only equal to r_e_minor for n=1 
        print('e: {0:.2f}'.format(cat.ellipticity.value))
        print('theta: {0:.2f}'.format(cat.orientation.value /np.pi))

        # --- measure curve of growth

        centre = (ndim/2., ndim/2.)        
        radii_pix = np.arange(1.0, 50., 1.0)
        radii_kpc = radii_pix * img.pixel_scale_kpc
        apertures = [CircularAperture(centre, r=r) for r in radii_pix] #r in pixels
        phot_table = aperture_photometry(data, apertures) 

        flux = np.array([phot_table['aperture_sum_{0}'.format(i)][0] for i, r in enumerate(radii_pix)])

        r_e = np.interp(0.5, flux/np.sum(data), radii_kpc)

        print('r_e: {0:.2f}'.format(r_e))
    
        r_e_major = r_e / np.sqrt(1 - cat.ellipticity.value)
    
        print('r_e_major: {0:.2f}'.format(r_e_major))
    


    
        
    