

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

import FLARE
import FLARE.filters
import SynthObs
from SynthObs.SED import models

import SynthObs.Morph.images 
import SynthObs.Morph.PSF 

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).sum(-1).sum(1)
    
    
    


filter = 'HST.WFC3.f160w'
F = FLARE.filters.add_filters([filter]) 
PSF = SynthObs.Morph.PSF.PSF(filter) # creates a dictionary of instances of the webbPSF class
flux = 1.


# ----- BASIC TEST OF ALL THE DIFFERENT IMAGES PRODUCED

do_test1 = False

if do_test1:

    width_arcsec = 2. # "
    
    imgs = SynthObs.Morph.images.point(flux, filter, width_arcsec, verbose = True, PSF = PSF)

    npanels = 5
    fig, axes = plt.subplots(1, npanels, figsize = (4*npanels,4))
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

    axes[0].imshow(imgs.super.hist, interpolation = 'nearest')
    axes[1].imshow(imgs.super.simple, interpolation = 'nearest')
    axes[2].imshow(imgs.super.data, interpolation = 'nearest')
    axes[3].imshow(imgs.img.data, interpolation = 'nearest')


    for ax in axes:    
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])


    # plt.savefig('test_images2.pdf')
    plt.show()
    

# ----- TEST OFFSETS

do_test2 = False

if do_test2:

    width_arcsec = 2. # "
    
    imgs = SynthObs.Morph.images.point(flux, filter, width_arcsec, verbose = True, PSF = PSF, xoffset_pix = 0.0, yoffset_pix = 0.0)

    im1 = imgs.img.data
    
    imgs = SynthObs.Morph.images.point(flux, filter, width_arcsec, verbose = True, PSF = PSF, xoffset_pix = 1.0, yoffset_pix = 0.0)

    im2 = imgs.img.data
    
    imgs = SynthObs.Morph.images.point(flux, filter, width_arcsec, verbose = True, PSF = PSF, xoffset_pix = 0.5, yoffset_pix = 0.0)

    im3 = imgs.img.data
    
    R1 = im1 - im2
    R2 = im1 - im3

    npanels = 2
    fig, axes = plt.subplots(1, npanels, figsize = (4*npanels,4))
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

    axes[0].imshow(R1, interpolation = 'nearest')
    axes[1].imshow(R2, interpolation = 'nearest')


    for ax in axes:    
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])


    # plt.savefig('test_images2.pdf')
    plt.show()




# ----- TEST USING MULTIPLE FILTERS

do_test3 = False

if do_test3:

    width_arcsec = 2. # "
    
    filters = ['HST.WFC3.f105w','HST.WFC3.f125w','HST.WFC3.f160w']
    F = FLARE.filters.add_filters(filters) 
    PSFs = SynthObs.Morph.PSF.PSFs(filters) # creates a dictionary of instances of the webbPSF class
    fluxes = {f: 100. for f in filters}
    
    imgs =  SynthObs.Morph.images.points(fluxes, filters, width_arcsec, resampling_factor = False, pixel_scale = False, PSFs = PSFs, verbose = False)

    npanels = len(filters)
    fig, axes = plt.subplots(1, npanels, figsize = (4*npanels,4))
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)


    for i,f in enumerate(filters):
        axes[i].imshow(imgs[f].data, interpolation = 'nearest')


    for ax in axes:    
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])


    # plt.savefig('test_images2.pdf')
    plt.show()



# ----- TEST OF ENCLOSED ENERGY

do_test4 = True

if do_test4:


    # see http://www.stsci.edu/hst/wfc3/analysis/ir_ee

    from photutils import CircularAperture
    from photutils import aperture_photometry

    width_arcsec = 2 # "
    pixel_scale = 0.06 #
    
    filters = ['HST.ACS.f435w','HST.WFC3.f105w','HST.WFC3.f125w','HST.WFC3.f160w']
    F = FLARE.filters.add_filters(filters) 
    PSFs = SynthObs.Morph.PSF.PSFs(filters, sub=5, verbose = True, charge_diffusion = True) # creates a dictionary of instances of the webbPSF class

       

    for filter in filters:
    
        psf = PSFs[filter]
    
        print('-'*5, filter)
    
        imgs = SynthObs.Morph.images.point(1.0, filter, width_arcsec, verbose = False, pixel_scale = pixel_scale, PSF = PSFs[filter], super_sampling = 5)
        
        img = imgs.img
        
        # --- ensquared energy
                    
        cx = img.ndim//2   
        print('central pixel: {0}'.format(cx))     
        for i in range(3):    
            print('{0}: {1:.2f}'.format(i*2+1, np.sum(img.data[cx-i:cx+i+1, cx-i:cx+i+1])))      

        
        # --- encircled energy
        
        centre = (img.ndim//2, img.ndim//2)   
        radii_arcsec = np.array([0.15, 0.5])
        radii_sampled_pix = radii_arcsec/(img.pixel_scale)
        apertures = [CircularAperture(centre, r=r) for r in radii_sampled_pix] #r in pixels
        phot_table = aperture_photometry(img.data, apertures) 
    
        for i in range(2): print('r={0}" f={1:.2f}'.format(radii_arcsec[i], phot_table[0][3+i]))
    
        
    




