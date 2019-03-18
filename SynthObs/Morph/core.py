from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

import webbpsf
from astropy.io import fits

from ..core import * 

import FLARE.filters 



def physical_images(X, Y, luminosities, filters, resolution = 0.1, Ndim = 100, smoothing = 'simple'):

    return {f: physical_image(X, Y, luminosities[f], resolution = resolution, Ndim = Ndim, smoothing = smoothing) for f in filters}
    

class physical_image():

    def __init__(self, X, Y, L, resolution = 0.1, Ndim = 100, smoothing = 'simple'):

        # Centre star particle positions using the median as the centre *** NOTE: true centre could later be defined ***
        X -= np.median(X)
        Y -= np.median(Y)

        # Boolean = Whether to apply gaussian smoothing to star particles
        self.smoothing = smoothing

        # Image properties
        self.Ndim = Ndim
        self.resolution = resolution
        self.width = Ndim * resolution 

        range = [np.max(X) - np.min(X), np.max(Y) - np.min(Y)]

        if any(x>Ndim*resolution for x in range): print('Warning particles will extend beyond image limits')

        

        self.img = np.zeros((self.Ndim, self.Ndim))

        # --- exclude particles not inside the image area
        
        sel = (np.fabs(X)<self.width/2.)&(np.fabs(Y)<self.width/2.)
        
        X = X[sel]
        Y = Y[sel]
        L = L[sel]

        
        if self.smoothing == 'simple_old':
        
            Gx, Gy = np.meshgrid(np.linspace(-self.width/2., self.width/2., Ndim), np.linspace(-self.width/2., self.width/2., Ndim))
        
            # --- can probably replace this with the unsmoothed but convolved by a gaussian

            r = 0.1
            
            for x,y,l in zip(X, Y, L):
                
                gauss = np.exp(-(((Gx - x)**2 + (Gy - y)**2)/ ( 2.0 * r**2 ) ) )  
                
                sgauss = np.sum(gauss)

                if sgauss > 0: self.img += l*gauss/sgauss


        elif self.smoothing == 'simple':
        
            Gx, Gy = np.meshgrid(np.linspace(-(self.width+self.resolution)/2., (self.width+self.resolution)/2., Ndim+1), np.linspace(-(self.width+self.resolution)/2., (self.width+self.resolution)/2., Ndim+1))
        
            r = 0.1
            gauss = np.exp(-((Gx**2 + Gy**2)/ ( 2.0 * r**2 ) ) )  
            gauss /= np.sum(gauss)
            
            g = np.linspace(-self.width/2.,self.width/2.,Ndim)
        
            for x,y,l in zip(X, Y, L):
        
                i, j = (np.abs(g - x)).argmin(), (np.abs(g - y)).argmin()
        
                self.img[j,i] += l
                
            self.img = convolve_fft(self.img, gauss)
            
            

        elif self.smoothing == 'adaptive':
         
            Gx, Gy = np.meshgrid(np.linspace(-self.width/2., self.width/2., Ndim), np.linspace(-self.width/2., self.width/2., Ndim))

            tree = cKDTree(np.column_stack([X, Y]), leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)

            nndists, nninds = tree.query(np.column_stack([X, Y]), k=7, n_jobs=-1) # k = nth nearest neighbour
        
            for x,y,l,nndist in zip(X, Y, L, nndists):
    
                r = np.max([nndist[-1], self.resolution])
                
                gauss = np.exp(-(((Gx - x)**2 + (Gy - y)**2)/ ( 2.0 * r**2 ) ) )  

                sgauss = np.sum(gauss)

                if sgauss > 0: self.img += l*gauss/sgauss


        else:
        
        
            g = np.linspace(-self.width/2.,self.width/2.,Ndim)
        
            for x,y,l in zip(X, Y, L):
        
                i, j = (np.abs(g - x)).argmin(), (np.abs(g - y)).argmin()
        
                self.img[j,i] += l











def webbPSFs(filters, width, resampling_factor=1):

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
            print('filter not found')

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
    
            fn = FLARE_dir + 'data/PSF/Euclid/Oesch/EUC_NISP_PSF-Y-SOLAR-AOCS-3632-SC3_20161212T220137.4Z_01.00.fits'
        
        else:
        
            fn = FLARE_dir + 'data/PSF/Euclid/Oesch/EUC_NISP_PSF-{0}-SOLAR-AOCS-3632-SC3_20161212T221037.4Z_01.00.fits'.format(f.split('.')[-1])   
        
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


def observed_images(X, Y, fluxes, filters, cosmo, redshift=8, width=10., resampling_factor=1, smoothed=True, PSFs = False, show = False):

    return {f: observed_image(X, Y, fluxes[f], f, cosmo, redshift, width, resampling_factor, smoothed, PSFs[f], show) for f in filters}
    

class observed_image():
    """ A class for computing synthetic Webb observations. Optionally applying gaussian smoothing (7th neighbour) to the
    initial image and if desired applying a PSF for the defined filter.
    """
    
    def __init__(self, X, Y, flux, filter, cosmo, redshift=8, width=10., resampling_factor=1, smoothed=True, PSF = None, show = False):
        """
        :param X: Star Particle X position in kpc. [nStar]
        :param Y: Star Particle Y position in kpc. [nStar]
        :param flux: An array of flux for each star particle for each filter in nJy. [nStar, nnircfilter]
        :param nircfilter: Either a string of the form JWST.NIRCam.XXXXX, where XXXXX is the desired filter code
        or the FWHM of the gaussian PSF (float).
        :param cosmo: A astropy.cosmology object.
        :param redshift: The redshift (z).
        :param width: Width of the image along a single axis (this is approximate since images must be odd in dimension)
        :param resampling_factor: The integer amount of resampling done to increase resolution. (int)
        :param smoothed: Boolean, whether to apply smoothing.
        :param PSF: Instance of the webbPSFs object for the desired filter .
        :param show: Boolean, whether to show images.
        """

        if filter in FLARE.filters.NIRCam_l: 
            self.resolution = 0.063
        elif filter in FLARE.filters.NIRCam_s: 
            self.resolution = 0.031
        elif filter in FLARE.filters.Euclid:
            self.resolution = 0.30
        else:
            print('filter not found')


        # Define instance attributes
        # Centre star particle positions using the median as the centre *** NOTE: true centre could later be defined ***
        self.X = X - np.median(X)
        self.Y = Y - np.median(Y)
        
        self.PSF = PSF # PSF object

        # Compute angular star particle positions in arcseconds
        
        self.arcsec_per_proper_kpc = cosmo.arcsec_per_kpc_proper(redshift).value
        
        self.X_arcsec = self.X * self.arcsec_per_proper_kpc
        self.Y_arcsec = self.Y * self.arcsec_per_proper_kpc
        
        self.cosmo = cosmo

        inst = filter.split('.')[1]
        f = filter.split('.')[-1]
        
        self.resampling_factor = resampling_factor
        self.base_pixel_scale = self.resolution  # pre-resample resolution
        self.pixel_scale = self.base_pixel_scale / self.resampling_factor  # the final image pixel scale
        self.smoothed = smoothed

        # Ndim must odd for convolution with the PSF
        ini_Ndim = int(width / self.resolution)
        if ini_Ndim % 2 != 0:
            self.Ndim = int(width / self.resolution)
        else:
            self.Ndim = int(width / self.resolution) + 1


        self.width = self.Ndim * self.pixel_scale  # width along each axis image in arcseconds
        self.flux = flux

        # if self.PSF is not None:
        #     assert self.Ndim == self.PSF.Ndim, 'PSF object must have the same dimensions as image object'
        #
        #     assert self.Ndim == self.PSF_dict.Ndim, 'PSF object must have the same dimensions as image object'
        #
        # # Make sure fluxes is compatible with the number of provided filters
        # if isinstance(filters, list):
        #     try:
        #         assert fluxes.shape[1] == len(filters), 'Fluxes must be provided for each filter'
        #     except IndexError:
        #         print('Fluxes must be provided for each filter')

        # Get the range of x and y star particle positions
        pos_range = [np.max(X) - np.min(X), np.max(Y) - np.min(Y)]

        # If star particles extend beyond the image print a warning
        if any(x > self.Ndim * self.pixel_scale for x in pos_range):
            print('Warning particles will extend beyond image limits')

        # If smoothing is required compute the smoothed image for each filter
        if self.smoothed:

            print('Smoothing...')

            self.img = self.smoothimg(self.flux)

        # If smoothing is not required compute the simple images
        else:

            self.img = self.simpleimg(self.flux)


        if self.PSF is not None:

            print('Applying PSF...')
            print(self.Ndim)
            # astropy.convolve requires images have odd dimensions
            assert self.Ndim % 2 != 0, 'Image must have odd dimensions (Ndim must be odd)'
            
            self.psf_img = self.psfimg()

        # If image output is required create and draw a quick plot for each image
        if show:

            print('Showing...')

            # If PSFs have been provided show simple images and PSF'd images
            if self.PSF:

                plt.figure(1)
                plt.imshow(self.img)
                plt.figure(2)
                plt.imshow(self.psf_img)

            # If no PSFs have been provided just show the simple images
            else:

                plt.figure(1)
                plt.imshow(self.img)

            plt.show()
        
    def simpleimg(self, F):
        """ A method for creating simple images where the stars are binned based on their position.

        :param F: The flux array for the current filter
        :return: Image array
        """

        # Initialise the image array
        simple_img = np.zeros((self.Ndim, self.Ndim))

        # Get the image pixel coordinates
        g = np.linspace(-self.width / 2., self.width / 2., self.Ndim)

        # Loop through star particles
        for x, y, l in zip(self.X_arcsec, self.Y_arcsec, F):

            # Get the stars position within the image
            i, j = (np.abs(g - x)).argmin(), (np.abs(g - y)).argmin()

            # Add the flux of this star to the corresponding pixel
            simple_img[j, i] += l

        return simple_img

    def smoothimg(self, F):
        """ A method for creating images with gaussian smoothing applied to each star with either the distance to the
        7th nearest neighbour or 0.1 kpc used for the standard deviation of the gaussian.

        :param F: The flux array for the current filter
        :return: Image array
        """

        # =============== Compute the gaussian smoothed image ===============

        # Define x and y positions for the gaussians
        Gx, Gy = np.meshgrid(np.linspace(-self.width / 2., self.width / 2., self.Ndim),
                             np.linspace(-self.width / 2., self.width / 2., self.Ndim))

        # Initialise the image array
        gsmooth_img = np.zeros((self.Ndim, self.Ndim))

        # Get the distances between all points using kdtree
        # Build tree
        tree = cKDTree(np.column_stack([self.X_arcsec, self.Y_arcsec]), leafsize=16, compact_nodes=True,
                       copy_data=False, balanced_tree=True)

        # Query tree for all particle separations
        nndists, nninds = tree.query(np.column_stack([self.X_arcsec, self.Y_arcsec]), k=7, n_jobs=-1)

        # Define the miniimum smoothing for 0.1kpc in arcseconds
        min_smooth = 0.1 * self.arcsec_per_proper_kpc

        # Get the image pixel coordinates along each axis
        ax_coords = np.linspace(-self.width / 2., self.width / 2., self.Ndim)

        rs = np.max([nndists[:, -1], np.full_like(self.X_arcsec, min_smooth)], axis=0)

        # Get the indices of stars with the minimum smoothing since these can be computed for a smaller sub image
        where_inds = np.where(rs == min_smooth)[0]

        # Loop over each star computing the smoothed gaussian distribution for this particle
        for x, y, l, r in zip(self.X_arcsec[where_inds], self.Y_arcsec[where_inds], F[where_inds], rs[where_inds]):

            # Get this star's position within the image
            x_img, y_img = (np.abs(ax_coords - x)).argmin(), (np.abs(ax_coords - y)).argmin()

            # Define sub image over which to compute the smooothing for this star (1/4 of the images size)
            # NOTE: this drastically speeds up image creation
            sub_xlow, sub_xhigh = x_img - int(self.Ndim / 8), x_img + int(self.Ndim / 8) + 1
            sub_ylow, sub_yhigh = y_img - int(self.Ndim / 8), y_img + int(self.Ndim / 8) + 1

            # Compute the image
            g = np.exp(-(((Gx[sub_xlow:sub_xhigh, sub_ylow:sub_yhigh] - x) ** 2
                          + (Gy[sub_xlow:sub_xhigh, sub_ylow:sub_yhigh] - y) ** 2)
                         / (2.0 * r ** 2)))

            # Get the sum of the gaussian
            gsum = np.sum(g)

            # If there are stars within the image in this gaussian add it to the image array
            if gsum > 0:
                gsmooth_img[sub_xlow:sub_xhigh, sub_ylow:sub_yhigh] += g * l / gsum

        # Get the indices of stars with greater than the minimum smoothing since these need
        # to be computed over the full image to account for the increased smoothing
        where_inds = np.where(rs != min_smooth)[0]

        # Loop over each star computing the smoothed gaussian distribution for this particle
        for x, y, l, r in zip(self.X_arcsec[where_inds], self.Y_arcsec[where_inds], F[where_inds], rs[where_inds]):

            # Compute the image
            g = np.exp(-(((Gx - x) ** 2 + (Gy - y) ** 2) / (2.0 * r ** 2)))

            # Get the sum of the gaussian
            gsum = np.sum(g)

            # If there are stars within the image in this gaussian add it to the image array
            if gsum > 0:
                gsmooth_img += g * l / gsum

        return gsmooth_img

    def psfimg(self):
        """ A method for applying the relevant PSF to an image made for a specific NIRCam filter.

        :param f: The filter string 'JWST.NIRCam.XXXXX'
        :return: The PSF'd image array.
        """

        # Convolve the PSF with the image
        convolved_img = convolve_fft(self.img, self.PSF.PSF)

        return convolved_img
