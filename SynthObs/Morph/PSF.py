
import numpy as np
from astropy.io import fits

from ..core import * 
import FLARE.filters 



def webbPSFs(filters, width, resampling_factor=1):

    import webbpsf

    return {f: webbPSF(f, width, resampling_factor) for f in filters}

     
class webbPSF():

    """ A class for extracting and storing point spread functions (PSFs) for the Webb Space
    Telescope using WebbPSF (STScI: https://webbpsf.readthedocs.io/en/stable/index.html).
    """

    def __init__(self, filter, width, resampling_factor=1):
        """
        :param f: tje filter of the form JWST.NIRCam.XXX or JWST.MIRI.XXX
        :param width: Width of the image along a single axis (this is approximate since images must be odd in dimension)
        :param resampling_factor: The integer amount of resampling done to increase resolution. (int)
        :param gaussFWHM: If a simple gaussian PSF is required the FWHM of that PSF in arcseconds. (float)
        """
        
        inst = filter.split('.')[1]
        f = filter.split('.')[-1]
        
        if filter in FLARE.filters.NIRCam_l: 
            self.resolution = 0.063
        elif filter in FLARE.filters.NIRCam_s: 
            self.resolution = 0.031
        else:
            print('filter not found', filter)

        # Ndim must odd for convolution with the PSF
        ini_Ndim = int(width / self.resolution)
        if ini_Ndim % 2 != 0:
            self.Ndim = int(width / self.resolution)
        else:
            self.Ndim = int(width / self.resolution) + 1

        if inst == 'NIRCAM':  nc = webbpsf.NIRCam()

        nc.filter = f
        self.PSF = nc.calc_psf(oversample=resampling_factor, fov_pixels=self.Ndim)[0].data  # compute PSF



def euclidPSFs(filters):

    return {f: euclidPSF(f) for f in filters}

class euclidPSF():

    def __init__(self, f):
    
        # --- these are only 85 * 85 but that is large!
    
        if f.split('.')[-1]=='Y':
    
            fn = FLARE_dir + '/data/PSF/Euclid/Oesch/EUC_NISP_PSF-Y-SOLAR-AOCS-3632-SC3_20161212T220137.4Z_01.00.fits'
        
        else:
        
            fn = FLARE_dir + '/data/PSF/Euclid/Oesch/EUC_NISP_PSF-{0}-SOLAR-AOCS-3632-SC3_20161212T221037.4Z_01.00.fits'.format(f.split('.')[-1])   
        
        self.PSF = fits.open(fn)[0].data



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

