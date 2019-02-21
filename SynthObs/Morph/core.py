from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.cosmology import WMAP9 as cosmos
import os
os.environ['WEBBPSF_PATH'] = '/Users/willroper/anaconda3/envs/webbpsf-env/share/webbpsf-data/'
os.environ['PYSYN_CDBS'] = '/Users/willroper/anaconda3/envs/webbpsf-env/share/pysynphot-data/cdbs/'
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

    def __init__(self, filters, width, resolution, resampling_factor=1, gaussFWHM=None):
        """

        :param filters: A string (or list of strings) of the form JWST.NIRCam.XXXXX, where X is the desired filter name.
        :param width: Width of the image along a single axis (this is approximate since images must be odd in dimension)
        :param resolution: The detector angular resolution in arcsecond per pixel (short wavelength channel= 0.031,
        long wavelength channel= 0.063)
        :param resampling_factor: The integer amount of resampling done to increase resolution. (int)
        :param gaussFWHM: If a simple gaussian PSF is required the FWHM of that PSF in arcseconds. (float)
        """

        # Define Attribute
        self.filters = filters  # filter list
        self.PSFs = {}  # PSF dicitonary. {filter name: PSF array}

        # Ndim must odd for convolution with the PSF
        ini_Ndim = int(width / resolution)
        if ini_Ndim % 2 != 0:
            self.Ndim = int(width / resolution)
        else:
            self.Ndim = int(width / resolution) + 1

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

    def __init__(self, X, Y, fluxes, filters, cosmo, redshift=8, width=10., resolution=0.031, resampling_factor=1,
                 smoothed=True, PSFs=None, show=False):
        """

        :param X: Star Particle X position in kpc. [nStar]
        :param Y: Star Particle Y position in kpc. [nStar]
        :param fluxes: An array of fluxes for each star particle for each filter in nJy. [nStar, nFilters]
        :param filters: A string (or list of strings) of the form JWST.NIRCam.XXXXX, where X is the desired filter name.
        :param cosmo: A astropy.cosmology object.
        :param redshift: The redshift (z).
        :param width: Width of the image along a single axis (this is approximate since images must be odd in dimension)
        :param resolution: The detector angular resolution in arcsecond per pixel (short wavelength channel= 0.031,
        long wavelength channel= 0.063)
        :param resampling_factor: The integer amount of resampling done to increase resolution. (int)
        :param smoothed: Boolean, whether to apply smoothing.
        :param PSFs: Instance of the webbPSFs object for the desired filters in filters or None.
        :param show: Boolean, whether to show images.
        """

        # Define instance attributes
        # Centre star particle positions using the median as the centre *** NOTE: true centre could later be defined ***
        self.X = X - np.median(X)
        self.Y = Y - np.median(Y)

        # Compute angular star particle positions in arcseconds
        self.X_arcsec = self.X * cosmo.arcsec_per_kpc_proper(redshift).value
        self.Y_arcsec = self.Y * cosmo.arcsec_per_kpc_proper(redshift).value

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
        self.fluxes = fluxes
        self.PSF_dict = PSFs  # PSF dictionary {filter name: PSF array}
        self.img = {}  # initialise the dictionary for created images

        # Make sure Ndim agrees with the PSF object that has been passed
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

            # If there is only one filter compute the single image
            if isinstance(filters, str):

                self.img[filters] = self.smoothimg(self.fluxes)

            # If there is more than one filter compute an image for each filter
            elif isinstance(filters, list):

                for f_ind in range(len(filters)):

                    self.img[filters[f_ind]] = self.smoothimg(self.fluxes[:, f_ind])

        # If smoothing is not required compute the simple images
        else:

            # If there is only one filter compute the single image
            if isinstance(filters, str):

                self.img[filters] = self.simpleimg(self.fluxes)

            # If there is more than one filter compute an image for each filter
            elif isinstance(filters, list):

                for f_ind in range(len(filters)):
                    self.img[filters[f_ind]] = self.simpleimg(self.fluxes[:, f_ind])

        # If PSFs have been provided (PSFs variable is an instance of webbPSFs class) apply the PSF for each filter
        if isinstance(PSFs, webbPSFs):

            print('Applying PSF...')
            print(self.Ndim)
            # astropy.convolve requires images have odd dimensions
            assert self.Ndim % 2 != 0, 'Image must have odd dimensions (Ndim must be odd)'

            # Initialise the dictionary to store the images with the PSF
            self.psf_img = {}

            # If filters is a list apply each PSF to each image for each filter
            if isinstance(filters, list):

                for NIRCf in filters:

                    self.psf_img[NIRCf] = self.psfimg(NIRCf)

            # If there is only one filter apply the PSF to the image
            elif isinstance(filters, str):

                self.psf_img[filters] = self.psfimg(filters)

        # If image output is required create and draw a quick plot for each image
        if show:

            print('Showing...')

            # If PSFs have been provided show simple images and PSF'd images
            if isinstance(PSFs, webbPSFs):

                # If there are multiple filters show each image
                if isinstance(filters, list):

                    for i, NIRCf in enumerate(filters):

                        plt.figure(i + 1)
                        plt.imshow(self.img[NIRCf])
                        plt.figure(len(filters) + (i + 1))
                        plt.imshow(self.psf_img[NIRCf])

                # If there is only a single filter show those images
                elif isinstance(filters, str):

                    plt.figure(1)
                    plt.imshow(self.img[filters])
                    plt.figure(2)
                    plt.imshow(self.psf_img[filters])

            # If no PSFs have been provided just show the simple images
            else:

                # If there are multiple filters show each one
                if isinstance(filters, list):

                    for i, NIRCf in enumerate(filters):
                        plt.figure(i + 1)
                        plt.imshow(self.img[NIRCf])

                # If there is only one filter show the single image
                elif isinstance(filters, str):

                    plt.figure(1)
                    plt.imshow(self.img[filters])

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

    def psfimg(self, f):
        """ A method for applying the relevant PSF to an image made for a specific NIRCam filter.

        :param f: The filter string 'JWST.NIRCam.XXXXX'
        :return: The PSF'd image array.
        """

        # Convolve the PSF with the image
        convolved_img = convolve(self.img[f], self.PSF_dict.PSFs[f])

        return convolved_img

# Define the variables needed to create comparison images
redshift = 8  # redshift
arc_res = 0.031
width = 6.  # img width in arcsec
# fs = ['JWST.NIRCam.F150W', 'JWST.NIRCam.F200W']
fs = ['JWST.NIRCam.F150W']

psfs = webbPSFs(fs, width, arc_res, gaussFWHM=4)

# Extract the x and y positions of stars in kpc/h
X = np.load('/Users/willroper/Documents/University/JWST/webster/data/086/234/3/X.npy')
Y = np.load('/Users/willroper/Documents/University/JWST/webster/data/086/234/3/Y.npy')

Ls = np.zeros((X.size, len(fs)))
for ind, f in enumerate(fs):
    # Extract the luminosity for the desired filter
    L = np.load('/Users/willroper/Documents/University/JWST/webster/data/086/234/3/'
                'ObservedLuminosities/BPASSv2.1.binary_ModSalpeter_300/' + f + '_default.npy')
    Ls[:, ind] = L

img = observed_image(X, Y, Ls, fs, cosmos, redshift=redshift, width=width, resolution=arc_res, resampling_factor=1,
                     smoothed=True, PSFs=psfs, show=True)
