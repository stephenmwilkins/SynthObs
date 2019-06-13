

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from photutils import CircularAperture
from photutils import aperture_photometry

import SynthObs.Morph.images 
import SynthObs.Morph.PSF 


# see http://www.stsci.edu/hst/wfc3/analysis/ir_ee

filter = 'HST.WFC3.f160w'


print('-'*5, filter)

PSF = SynthObs.Morph.PSF.PSF(filter) # creates a dictionary of instances of the webbPSF class

show = False

if show:
    plt.imshow(np.log10(PSF.data))
    plt.show()

    plt.imshow(np.log10(PSF.convolved_data))
    plt.show()

Ndim = PSF.convolved_data.shape[0]

x = y = np.linspace(-(Ndim/2.)/5, (Ndim/2.)/5, Ndim) # in original pixels

if show:
    plt.imshow(np.log10(PSF.f(x,y)))
    plt.show()



centre = (Ndim//2, Ndim//2)

radii_arcsec = np.array([0.15, 0.5])
radii_sampled_pix = radii_arcsec/(0.13/5.)
apertures = [CircularAperture(centre, r=r) for r in radii_sampled_pix] #r in pixels


for img, label in zip([PSF.data, PSF.convolved_data, PSF.f(x,y)],['raw', 'convolved', 'functional']):
    print('-'*5, label)
    phot_table = aperture_photometry(img, apertures) 
    for i in range(2): print('r={0}" f={1:.2f}'.format(radii_arcsec[i], phot_table[0][3+i]))





width_arcsec = 2. # "
    
print('-'*5, 'in image')
    
observed, super = SynthObs.Morph.images.point(1.0, filter, width_arcsec, pixel_scale = 0.06, verbose = False, PSF = PSF)

img = super

Ndim = img.img.shape[0]
centre = (Ndim//2, Ndim//2)

radii_arcsec = np.array([0.15, 0.5])
radii_sampled_pix = radii_arcsec/(img.pixel_scale)
apertures = [CircularAperture(centre, r=r) for r in radii_sampled_pix] #r in pixels
phot_table = aperture_photometry(img.img, apertures) 
for i in range(2): print('r={0}" f={1:.2f}'.format(radii_arcsec[i], phot_table[0][3+i]))



