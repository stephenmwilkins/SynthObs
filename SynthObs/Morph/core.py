from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel
import webbpsf


class physical_image():

    def __init__(self, X, Y, L, resolution = 0.1, Ndim = 100, smoothed = True, show = False):

        # Centre star particle positions using the median as the centre *** NOTE: true centre could later be defined ***
        X -= np.median(X)
        Y -= np.median(Y)

        # Boolean = Whether to apply gaussian smoothing to star particles
        self.smoothed = smoothed

        # Image properties
        self.Ndim = Ndim
        self.resolution = resolution
        self.width = Ndim * resolution 

        range = [np.max(X) - np.min(X), np.max(Y) - np.min(Y)]
        print(range) 

        if any(x>Ndim*resolution for x in range): print('Warning particles will extend beyond image limits')

        if not self.smoothed: g = np.linspace(-self.width/2.,self.width/2.,Ndim)

        if self.smoothed: Gx, Gy = np.meshgrid(np.linspace(-self.width/2., self.width/2., Ndim),
                                               np.linspace(-self.width/2., self.width/2., Ndim))

        self.img = np.zeros((self.Ndim, self.Ndim))

        if self.smoothed:

            tree = cKDTree(np.column_stack([X, Y]), leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)

            nndists, nninds = tree.query(np.column_stack([X, Y]), k=7, n_jobs=-1) # k = nth nearest neighbour
        
            for x,y,l,nndist in zip(X, Y, L, nndists):
    
                r = np.max([nndist[-1], 0.1])
    
                gauss = np.exp(-(((Gx - x)**2 + (Gy - y)**2)/ ( 2.0 * r**2 ) ) )  

                sgauss = np.sum(gauss)

                if sgauss > 0: self.img += l*gauss/sgauss

        else:
        
            for x,y,l in zip(X, Y, L):
        
                i, j = (np.abs(g - x)).argmin(), (np.abs(g - y)).argmin()
        
                self.img[j,i] += l
        
        if show:
        
            plt.imshow(self.img)
            plt.show()

     
class webbPSFs():
    """ A class for extracting and storing point spread functions (PSFs) for the Webb Space
    Telescope using WebbPSF (STScI: https://webbpsf.readthedocs.io/en/stable/index.html).
    """

    def __init__(self, nircfilter, width, resolution, resampling_factor=1):
        """
        :param nircfilter: Either a string of the form JWST.NIRCam.XXXXX, where XXXXX is the desired filter code
        or the FWHM of the gaussian PSF (float).
        :param width: Width of the image along a single axis (this is approximate since images must be odd in dimension)
        :param resolution: The detector angular resolution in arcsecond per pixel (short wavelength channel= 0.031,
        long wavelength channel= 0.063)
        :param resampling_factor: The integer amount of resampling done to increase resolution. (int)
        :param gaussFWHM: If a simple gaussian PSF is required the FWHM of that PSF in arcseconds. (float)
        """

        # Define Attribute
        self.nircfilter = nircfilter  # filter list

        # Ndim must odd for convolution with the PSF
        ini_Ndim = int(width / resolution)
        if ini_Ndim % 2 != 0:
            self.Ndim = int(width / resolution)
        else:
            self.Ndim = int(width / resolution) + 1

        # If nircfilter is a valid webbPSF
        if isinstance(nircfilter, str):
            
            # Compute the PSF
            nc = webbpsf.NIRCam()  # Assign NIRCam object to variable.
            nc.filter = self.nircfilter.split('.')[-1]  # Set filter.
            self.PSF = nc.calc_psf(oversample=resampling_factor, fov_pixels=self.Ndim)[0].data  # compute PSF

        # If no gaussian FWHM is passed use Web PSFs
        if gaussFWHM is None:

            # Check if filters is a list or string
            # If filters is a list create a PSF for each filter
            if isinstance(filters, list):

                for fstring in self.filters:

                    f = fstring.split('.')[-1]
                    # Compute the PSF
                    nc = webbpsf.NIRCam()  # Assign NIRCam object to variable.
                    nc.filter = f  # Set filter.
                    self.PSFs[fstring] = nc.calc_psf(oversample=resampling_factor, fov_pixels=self.Ndim)[0].data  # compute PSF

            # If it is a string create a single PSF for that string
            elif isinstance(filters, str):

                # Compute the PSF
                nc = webbpsf.NIRCam()  # Assign NIRCam object to variable.
                nc.filter = self.filters.split('.')[-1]  # Set filter.
                self.PSFs[self.filters] = nc.calc_psf(oversample=resampling_factor, fov_pixels=self.Ndim)[0].data  # compute PSF

            # If neither of the previous conditions are satisfied then filters is not an acceptable format
            else:
                print('WARNING: Incompatible format for filters, '
                      'should be list: [JWST.NIRCam.XXXXX] or string: "JWST.NIRCam.XXXXX" ')


        # Else create a gaussian PSF
        else:
            self.PSF = Gaussian2DKernel(nircfilter/2.355)
            # Check if filters is a list or string
            # If filters is a list create a PSF for each filter
            if isinstance(filters, list):

                for fstring in self.filters:

                    f = fstring.split('.')[-1]
                    self.PSFs[fstring] = Gaussian2DKernel(gaussFWHM/2.355)

            # If it is a string create a single PSF for that string
            elif isinstance(filters, str):

                # Compute the PSF
                nc = webbpsf.NIRCam()  # Assign NIRCam object to variable.
                nc.filter = self.filters.split('.')[-1]  # Set filter.
                self.PSFs[self.filters] = Gaussian2DKernel(gaussFWHM/2.355)

            # If neither of the previous conditions are satisfied then filters is not an acceptable format
            else:
                print('WARNING: Incompatible format for filters, '
                      'should be list: [JWST.NIRCam.XXXXX] or string: "JWST.NIRCam.XXXXX" ')




class observed_image():
    """ A class for computing synthetic Webb observations. Optionally applying gaussian smoothing (7th neighbour) to the
    initial image and if desired applying a PSF for the defined filter.
    """
    
    def __init__(self, X, Y, flux, nircfilter, cosmo, redshift=8, width=10., resolution=0.031, resampling_factor=1,
                 smoothed=True, PSF_obj=None, show=False):
        """
        :param X: Star Particle X position in kpc. [nStar]
        :param Y: Star Particle Y position in kpc. [nStar]
        :param flux: An array of flux for each star particle for each filter in nJy. [nStar, nnircfilter]
        :param nircfilter: Either a string of the form JWST.NIRCam.XXXXX, where XXXXX is the desired filter code
        or the FWHM of the gaussian PSF (float).
        :param cosmo: A astropy.cosmology object.
        :param redshift: The redshift (z).
        :param width: Width of the image along a single axis (this is approximate since images must be odd in dimension)
        :param resolution: The detector angular resolution in arcsecond per pixel (short wavelength channel= 0.031,
        long wavelength channel= 0.063)
        :param resampling_factor: The integer amount of resampling done to increase resolution. (int)
        :param smoothed: Boolean, whether to apply smoothing.
        :param PSF_obj: Instance of the webbPSFs object for the desired nircfilter in nircfilter or None.
        :param show: Boolean, whether to show images.
        """

        # Define instance attributes
        # Centre star particle positions using the median as the centre *** NOTE: true centre could later be defined ***
        self.X = X - np.median(X)
        self.Y = Y - np.median(Y)

        # Compute angular star particle positions in arcseconds
        
        self.arcsec_per_proper_kpc = cosmo.arcsec_per_kpc_proper(redshift).value
        
        self.X_arcsec = self.X * self.arcsec_per_proper_kpc
        self.Y_arcsec = self.Y * self.arcsec_per_proper_kpc
        
        self.cosmo = cosmo

        self.resampling_factor = resampling_factor
        self.base_pixel_scale = resolution  # pre-resample resolution
        self.pixel_scale = self.base_pixel_scale / self.resampling_factor  # the final image pixel scale
        self.smoothed = smoothed

        # Ndim must odd for convolution with the PSF
        ini_Ndim = int(width / self.pixel_scale)
        if ini_Ndim % 2 != 0:
            self.Ndim = int(width / self.pixel_scale)
        else:
            self.Ndim = int(width / self.pixel_scale) + 1

        self.width = self.Ndim * self.pixel_scale  # width along each axis image in arcseconds
        self.flux = flux
        self.PSF = PSF_obj  # PSF object

        assert self.Ndim == self.PSF.Ndim, 'PSF object must have the same dimensions as image object'

        assert self.Ndim == self.PSF_dict.Ndim, 'PSF object must have the same dimensions as image object'

        # Make sure fluxes is compatible with the number of provided filters
        if isinstance(filters, list):
            try:
                assert fluxes.shape[1] == len(filters), 'Fluxes must be provided for each filter'
            except IndexError:
                print('Fluxes must be provided for each filter')

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

        # If PSFs have been provided (PSFs variable is an instance of webbPSFs class) apply the PSF for each filter
        if isinstance(self.PSF, webbPSFs):

            print('Applying PSF...')
            print(self.Ndim)
            # astropy.convolve requires images have odd dimensions
            assert self.Ndim % 2 != 0, 'Image must have odd dimensions (Ndim must be odd)'
            
            self.psf_img = self.psfimg()

        # If image output is required create and draw a quick plot for each image
        if show:

            print('Showing...')

            # If PSFs have been provided show simple images and PSF'd images
            if isinstance(self.PSF, webbPSFs):

                plt.figure(1)
                plt.imshow(self.img)
                plt.figure(2)
                plt.imshow(self.psf_img)

            # If no PSFs have been provided just show the simple images
            else:

                plt.figure(1)
                plt.imshow(self.img)

            plt.show()
            
    @classmethod
    def multifilter(cls, X, Y, fluxes, nircfilters, cosmo, redshift=8, width=10., alt_res=0.01, resampling_factor=1,
                     smoothed=True, show=False):
        """ A helper method to create a set of images for the same object in different filters WITH PSF. If a gaussian
        PAF is required an alternative resolution must be supplied and nircfilters must be a list of the FWHMs for the
        gaussian kernel.

        :param X: Star Particle X position in kpc. [nStar]
        :param Y: Star Particle Y position in kpc. [nStar]
        :param fluxes: A dictionary of flux arrays for the star particles. {nircfilters:flux array}
        :param nircfilters: Either a list of strings of the form JWST.NIRCam.XXXXX, where XXXXX is the desired
        filter code or a list of the FWHM of the gaussian PSFs (float).
        :param cosmo: A astropy.cosmology object.
        :param redshift: The redshift (z).
        :param width: Width of the image along a single axis (this is approximate since images must be odd in dimension)
        :param alt_res: The non NIRCam resolution to use if non-Webb images are required. If using Webb filters at
        detector resolution this does not need to supplied.
        :param resampling_factor: The integer amount of resampling done to increase resolution. (int)
        :param smoothed: Boolean, whether to apply smoothing.
        :param show: Boolean, whether to show images.
        :return:
        """

        # For W, M and N, R=4,10,100 respectively (resolving power)
        # Define lists for filters associated with long wavelength observations (0.063" resolution)
        # and short wavelength observations (0.031").
        longFilters = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N+F322W2', 'F335M', 'F356W', 'F360M',
                       'F405N+F444W', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N+F444W', 'F470N+F444W', 'F480M']
        shortFilters = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2', 'F182M', 'F187N', 'F200W', 'F210M',
                        'F212N', 'F164N+F150W2', 'F162M+F150W2']

        # Check a flux has been provided for each filter
        assert len(fluxes.keys()) == len(nircfilters), 'Each filter must have a corresponding flux array'

        # Initialise image dictionaries
        imgs = {}
        psf_imgs = {}

        # Loop through filters
        for nircf in nircfilters:

            # Get the resolution for the supplied filter unless a gaussian is required and an alternative resolution
            # has been defined
            if nircf.split('.')[-1] in shortFilters:
                resolution = 0.031
            elif nircf.split('.')[-1] in longFilters:
                resolution = 0.063
            else:
                resolution = alt_res

            # Get the PSF for this filter
            PSF_obj = webbPSFs(nircf, width, resolution)

            # Initialise image object for this filter and flux combination
            ob_img = cls(X, Y, fluxes[nircf], nircf, cosmo, redshift, width, resolution, resampling_factor,
                     smoothed, PSF_obj, show)

            # Extract the simple image from the image object and store it in the dictionary
            imgs[nircf] = ob_img.img

            # Extract the image with psf from the image object and store it in the dictionary
            psf_imgs[nircf] = ob_img.psf_img

        return imgs, psf_imgs


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
        min_smooth = 0.1 * self.cosmo.arcsec_per_kpc_proper(redshift).value

        # Loop over each star computing the smoothed gaussian distribution for this particle
        for x, y, l, nndist in zip(self.X_arcsec, self.Y_arcsec, F, nndists):

            # If the 7th nn distance is less than 0.1 use 0.1
           
            r = max([nndist[-1], min_smooth])

            # Compute the image
            g = np.exp(-(((Gx - x) ** 2 + (Gy - y) ** 2) / (2.0 * r ** 2)))

            # Get the sum of the gaussians
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
        convolved_img = convolve(self.img, self.PSF.PSF)

        return convolved_img
