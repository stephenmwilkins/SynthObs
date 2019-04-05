
import numpy as np
from astropy.io import fits
from scipy import interpolate

from ..core import * 
import FLARE.filters 



def Webb(filters, resampling_factor = 5):

    return {f: WebbPSF(f, resampling_factor) for f in filters}

 
class WebbPSF():

    def __init__(self, f, resampling_factor = 5):
    
        self.filter = f
        self.inst = f.split('.')[1]
    
        fn = FLARE_dir + '/data/PSF/Webb/{1}/{0}/{2}.fits'.format(resampling_factor, self.inst, f.split('.')[-1])   
        
        self.data = fits.open(fn)[0].data

        Ndim = self.data.shape[0]

        x = y = np.linspace(-(Ndim/2.)/resampling_factor, (Ndim/2.)/resampling_factor, Ndim)

        self.f = interpolate.interp2d(x, y, self.data, kind='linear')
 

     
def Hubble(filters):

    return {f: HubblePSF(f) for f in filters}

 
class HubblePSF():

    def __init__(self, f):
    
        self.filter = f
        self.inst = f.split('.')[1]
    
        fn = FLARE_dir + '/data/PSF/Hubble/{0}/PSFSTD_WFC3IR_{1}.fits'.format(self.inst, f.split('.')[-1].upper())   
        
        self.data = fits.open(fn)[0].data[0]

        Ndim = self.data.shape[0]

        x = y = np.linspace(-12.5, 12.5, Ndim) # Hubble PSFs are oversampled

        self.f = interpolate.interp2d(x, y, self.data, kind='linear')




def Euclid(filters):

    return {f: EuclidPSF(f) for f in filters}

class EuclidPSF():

    def __init__(self, f, scale = '300mas'):
    
        self.filter = f
    
        fn = FLARE_dir + '/data/PSF/Euclid/Oesch/{0}/Euclid_PSF_{1}_{0}.fits'.format(scale, f.split('.')[-1])   
        
        self.data = fits.open(fn)[0].data

        if scale == '300mas': x = y = np.arange(-self.data.shape[0]/2.+0.5, self.data.shape[0]/2., 1.)

        self.f = interpolate.interp2d(x, y, self.data, kind='linear')


# class gauss():
# 
#     def __init__():
# 
#         self.PSF = Gaussian2DKernel(nircfilter/2.355)
#         # Check if filters is a list or string
#         # If filters is a list create a PSF for each filter
#         if isinstance(filters, list):
# 
#             for fstring in self.filters:
# 
#                 f = fstring.split('.')[-1]
#                 self.PSFs[fstring] = Gaussian2DKernel(gaussFWHM/2.355)
# 
#         # If it is a string create a single PSF for that string
#         elif isinstance(filters, str):
# 
#             # Compute the PSF
#             nc = webbpsf.NIRCam()  # Assign NIRCam object to variable.
#             nc.filter = self.filters.split('.')[-1]  # Set filter.
#             self.PSFs[self.filters] = Gaussian2DKernel(gaussFWHM/2.355)
# 

