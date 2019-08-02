

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from photutils import CircularAperture
from photutils import aperture_photometry

import SynthObs.Morph.images 
import SynthObs.Morph.PSF 












# https://hst-docs.stsci.edu/display/WFC3IHB/7.6+IR+Optical+Performance

filter = 'HST.WFC3.f160w'

print('-'*5, filter)

sub = 5

PSF = SynthObs.Morph.PSF.PSF(filter, sub=5) 

ndim = PSF.ndim

width = ndim/sub

x = y = np.linspace(-width/2, width/2., ndim) # in original pixels

psf = PSF.f(x,y)


# plt.imshow(np.log10(psf/np.max(psf)), vmin = -4, vmax = 0.0)
# plt.show()



g = np.linspace(-width/2, width/2., ndim) # in original pixels
xx, yy = np.meshgrid(g, g)  
PSF = SynthObs.Morph.PSF.gauss(1.176)
gauss = PSF.f(xx,yy)
gauss /= np.sum(gauss)



centre = (ndim//2, ndim//2)
radii_pix = np.arange(1,100,1)
apertures = [CircularAperture(centre, r=r) for r in radii_pix] #r in pixels

phot_table = aperture_photometry(psf, apertures) 
flux_psf = np.array([phot_table[0][3+i] for i in range(len(radii_pix))])
phot_table = aperture_photometry(gauss, apertures) 
flux_gauss = np.array([phot_table[0][3+i] for i in range(len(radii_pix))])




plt.plot(radii_pix[:-1], flux_psf[1:]-flux_psf[0:-1])
plt.plot(radii_pix[:-1], (flux_gauss[1:]-flux_gauss[0:-1])*0.5)
plt.show()



