

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import FLARE
import FLARE.filters
import SynthObs
import SynthObs.Morph.PSF 

from photutils import CircularAperture
from photutils import aperture_photometry

logged = False

base_width = 5. # "
sampling = 0.1

filters = ['Euclid.NISP.H', 'HST.WFC3.f160w', 'JWST.NIRCAM.F150W'] #

# filters = ['Spitzer.IRAC.ch1', 'JWST.NIRCAM.F356W'] #


print(filters)


PSFs = SynthObs.Morph.PSF.PSF(filters) 




fig2 = plt.figure()
ax2 = fig2.add_axes((0.15, 0.15, 0.8, 0.8))





fig, axes = plt.subplots(1, len(filters), figsize = (len(filters)*2., 2))
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

for ax, f in zip(axes.flatten(), filters):
    
    print('-'*10, f)
    
    width_pix_sampled = base_width // (FLARE.filters.pixel_scale[f] * sampling) # width of sampled pixels -> want to be odd
    
    if width_pix_sampled % 2 == 0: width_pix_sampled -= 1
        
    print(width_pix_sampled)
    
    width_pix = width_pix_sampled * sampling
    
    print(width_pix)
    
    xx = yy = np.linspace(-width_pix/2., width_pix/2. , width_pix_sampled)  # super-sampled to 0.1 * original pixels
    
    img = PSFs[f].f((xx-0.0), (yy-0.0))
    
    
    
    img /= np.sum(img)
    
    
    # --- plot 2D PSF
    
    if logged:
        ax.imshow(np.log10(img/np.max(img)), interpolation = 'nearest', vmin = -4, vmax = 0.0)
    else:           
        ax.imshow(img, interpolation = 'nearest')
     
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.text(0.5, 0.85, f.split('.')[-1], fontsize = 15, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


    # --- measure PSF profile


    Ndim = img.shape[0]

    centre = (Ndim//2, Ndim//2)

    radii_arcsec = np.arange(0.01, 3.1, 0.01)
    radii_sampled_pix = radii_arcsec/(FLARE.filters.pixel_scale[f]*sampling)

    apertures = [CircularAperture(centre, r=r) for r in radii_sampled_pix] #r in pixels
    
    phot_table = aperture_photometry(img, apertures) 
    
    frac = np.array([phot_table[0][3+i] for i in range(len(radii_arcsec))])
    # print(frac)
    print(frac.shape)
    print(radii_arcsec.shape)
    
    for efrac in [0.5, 0.8]:
    
        print('EE(f={0}): {1:0.2f}"'.format(efrac, np.interp(efrac, frac, radii_arcsec)))
    
    
    for r in [0.35/2.]:
    
        print('EE(r<{0}): {1:0.2f}'.format(r, np.interp(r, radii_arcsec, frac)))
    
    ax2.plot(radii_arcsec, frac, label = f)
    



fig.savefig('f/PSF1.pdf')
fig.clf()


fig2.legend()
fig2.savefig('f/PSF2.pdf')
fig2.clf()


