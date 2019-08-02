
import numpy as np
from astropy.io import fits
from scipy import interpolate

import matplotlib.pyplot as plt

from ..core import * 
import FLARE.filters 

from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

from astropy.modeling import models

from skimage.transform import resize




def PSFs(filters, **kwargs):
       
    return {f:PSF(f, **kwargs) for f in filters}


def PSF(f, **kwargs):
        
    if f.split('.')[0] == 'HST': psf = HubblePSF(f, **kwargs)
    if f.split('.')[0] == 'JWST': psf = WebbPSF(f, **kwargs)
    if f.split('.')[0] == 'Euclid': psf = EuclidPSF(f, **kwargs)
    if f.split('.')[0] == 'Spitzer': psf = SpitzerPSF(f, **kwargs)
    
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

    def __init__(self, f, sub = 5, charge_diffusion = True, verbose = False):
    
        if verbose: print('--- PSF')
        if verbose: print(f)
        if verbose: print('sub-sampling: {0}'.format(sub))
    
        self.filter = f
    
        fn = FLARE_dir + '/data/PSF/Hubble/TinyTim/sub{1}/{0}00.fits'.format(f.split('.')[-1], sub)   
                
        hdu = fits.open(fn)
        
        self.data = hdu[0].data[3:-2,3:-2]
        
        if verbose: print('sun(data): {0}'.format(np.sum(self.data)))

        if sub != 1: 
            
            self.charge_diffusion_kernel = np.zeros((5,5))
            self.charge_diffusion_kernel[1:-1,1:-1] = np.array([list(map(float, ' '.join(hdu[0].header[-(3-i)].split(' ')).split())) for i in range(3)])

        if charge_diffusion and sub != 1:
        
            resampled_charge_diffusion_kernel = resize(self.charge_diffusion_kernel, (5*sub, 5*sub), anti_aliasing = False)
            
            self.data = convolve_fft(self.data, resampled_charge_diffusion_kernel)

        self.ndim = self.data.shape[0]

        self.width = self.ndim*FLARE.filters.pixel_scale[f]/sub # "

        if verbose: print('ndim: {0}'.format(self.ndim))

        x = y = np.linspace(-(self.ndim/2.)/sub, (self.ndim/2.)/sub, self.ndim) # in original pixels

        self.f = interpolate.interp2d(x, y, self.data, kind='linear', fill_value = 0.0)


 
 
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










class gauss():

    def __init__(self, FWHM):

        self.stddev = FWHM/(2*np.sqrt(2*np.log(2)))
        
        self.f = models.Gaussian2D(amplitude=1.0, x_mean=0.0, y_mean=0.0, x_stddev=self.stddev, y_stddev=self.stddev)



# class gauss():
# 
#     def __init__(self, FWHM, sub = 5):
# 
# 
#         self.size = 11 # original pixels
#         self.FWHM = FWHM
#         self.std = (FWHM/2.355)*sub
#         self.data = Gaussian2DKernel(self.std, y_stddev=self.std, x_size = self.size*sub, y_size = self.size*sub, mode = 'linear_interp') # in pixels
#         
#         print(self.data.shape[0])
#         
#         ndim = self.data.shape[0]
#         
#         x = y = np.linspace(-(self.size/2.), (self.size/2.), ndim) # in original pixels
#         
#         self.f = interpolate.interp2d(x, y, self.data, kind='linear', fill_value = 0.0)
