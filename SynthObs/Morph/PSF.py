
import numpy as np
from astropy.io import fits
from scipy import interpolate

from ..core import * 
import FLARE.filters 

from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel


def PSFs(filters):
       
    return {f:PSF(f) for f in filters}


def PSF(f):
        
    if f.split('.')[0] == 'HST': psf = HubblePSF(f)
    if f.split('.')[0] == 'JWST': psf = WebbPSF(f)
    if f.split('.')[0] == 'Euclid': psf = EuclidPSF(f)
    if f.split('.')[0] == 'Spitzer': psf = SpitzerPSF(f)
    
    return psf


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

        self.f = interpolate.interp2d(x, y, self.data, kind='linear', fill_value = 0.0)
 

     
def Hubble(filters):

    return {f: HubblePSF(f) for f in filters}

 
 
class HubblePSF():

    def __init__(self, f):
    
        self.filter = f
    
        fn = FLARE_dir + '/data/PSF/Hubble/TinyTim/{0}00.fits'.format(f.split('.')[-1])   
        
        self.data = fits.open(fn)[0].data[1:,1:]

        charge_diffusion_kernel = np.array([[0.002, 0.038, 0.002],[0.038, 0.840, 0.038],[0.002, 0.038, 0.002]]) # pixels?
        
        x = y = np.linspace(-1, 1, 3) # in original pixels
        
        f_charge_diffusion_kernel= interpolate.interp2d(x, y, charge_diffusion_kernel, kind='linear', fill_value = 0.0)
        
        x = y = np.linspace(-3, 3, 31) 

        resampled_charge_diffusion_kernel = f_charge_diffusion_kernel(x,y)

        self.convolved_data = convolve(self.data, resampled_charge_diffusion_kernel)

        Ndim = self.convolved_data.shape[0]

        

        x = y = np.linspace(-(Ndim/2.)/5., (Ndim/2.)/5., Ndim) # in original pixels

        self.f = interpolate.interp2d(x, y, self.convolved_data, kind='linear', fill_value = 0.0)


 
 
def Euclid(filters):

    return {f: EuclidPSF(f) for f in filters}

class EuclidPSF():

    def __init__(self, f, scale = '50mas'):
    
        self.filter = f
    
        fn = FLARE_dir + '/data/PSF/Euclid/Oesch/{0}/Euclid_PSF_{1}_{0}.fits'.format(scale, f.split('.')[-1])   
        
        self.data = fits.open(fn)[0].data

        self.data /= np.sum(self.data)

        Ndim = self.data.shape[0]

        if scale == '300mas': x = y = np.arange(-self.data.shape[0]/2.+0.5, self.data.shape[0]/2., 1.)
        if scale == '50mas': x = y = np.linspace(-Ndim/12., Ndim/12., Ndim)

        self.f = interpolate.interp2d(x, y, self.data, kind='linear', fill_value = 0.0)



def Spitzer(filters):

    return {f: SpitzerPSF(f) for f in filters}

class SpitzerPSF():

    def __init__(self, f):
    
        self.filter = f
    
        if f.split('.')[-1] == 'ch1': self.FWHM = 1.66 # " (1.95 for warm)
        if f.split('.')[-1] == 'ch2': self.FWHM = 1.72 # " (2.02 for warm)
        
        self.FWHM /= 1.22 # in native pixels

    def f(self, x, y):
    
        xx, yy = np.meshgrid(x, y)

        return np.exp(-4*np.log(2) * (xx**2 + yy**2) / self.FWHM**2)











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

