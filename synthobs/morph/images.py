from scipy.spatial import cKDTree
import numpy as np

import matplotlib.pyplot as plt
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.modeling.models import Sersic2D

from astropy.io import fits

from ..core import *

import flare.filters
import flare.observatories

class empty(): pass



def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).sum(-1).sum(1)


# create an image with no PSF smoothing. This is a base class to be called by other routines.


def core(X, Y, L, resolution = 0.1, ndim = 100, smoothing = False, verbose = False):

    # X, Y, resolution, and smoothing scale all in the same units (default physical kpc)

    img = empty()

    if smoothing:
        smoothing_method, smoothing_scale = smoothing
    else:
        smoothing_method = False

    width = ndim * resolution

    img.hist = np.zeros((ndim, ndim))
    img.simple = np.zeros((ndim, ndim))
    img.data = np.zeros((ndim, ndim))

    # --- exclude particles not inside the image area

    sel = (np.fabs(X)<width/2.)&(np.fabs(Y)<width/2.)
    X = X[sel]
    Y = Y[sel]
    L = L[sel]


    G = np.linspace(-width/2.,width/2., ndim)
    # G = np.linspace(-(width+resolution)/2., (width+resolution)/2., ndim+1)
    Gx, Gy = np.meshgrid(G, G)


    if verbose:
        print('*'*5, 'CORE')
        print('width: {0:.2f} kpc'.format(width))
        print('ndim: {0:.2f} '.format(ndim))
        print('resolution: {0:.2f} '.format(resolution))
        print('N_particles: {0:.2f} '.format(len(X)))
        print('smoothing method: {0} '.format(smoothing_method))

    for x,y,l in zip(X, Y, L):
        i, j = (np.abs(G - x)).argmin(), (np.abs(G - y)).argmin()
        img.simple[j,i] += l
        img.hist[j,i] += 1


    if smoothing_method == 'convolved_gaussian':

        sigma = smoothing_scale/2.355
        gauss = np.exp(-((Gx**2 + Gy**2)/ ( 2.0 * sigma**2 ) ) )
        gauss /= np.sum(gauss)

        img.data = convolve_fft(img.simple, gauss)


    elif smoothing_method == 'adaptive':

        tree = cKDTree(np.column_stack([X, Y]), leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)

        nndists, nninds = tree.query(np.column_stack([X, Y]), k = smoothing_scale, n_jobs=-1) # k = nth nearest neighbour

        for x,y,l,nndist in zip(X, Y, L, nndists):

            FWHM = np.max([nndist[-1], 0.01])

            sigma = FWHM/2.355

            gauss = np.exp(-(((Gx - x)**2 + (Gy - y)**2)/ ( 2.0 * sigma**2 ) ) )

            sgauss = np.sum(gauss)

            if sgauss > 0: img.data += l*gauss/sgauss



    else:

        img.data = img.simple

    return img






class observed():

    def __init__(self, filter, cosmo, z, target_width_arcsec, resampling_factor=False, pixel_scale=False, smoothing = False, PSF = False, super_sampling = 4, verbose = False, xoffset_pix = 0.0, yoffset_pix = 0.0):

        self.filter = filter
        self.target_width_arcsec = target_width_arcsec
        self.resampling_factor = resampling_factor
        self.pixel_scale = pixel_scale
        self.smoothing = smoothing
        self.PSF = PSF
        self.verbose = verbose
        self.super_sampling = super_sampling

        self.native_pixel_scale = flare.observatories.filter_info[self.filter]['pixel_scale']

        if self.resampling_factor:
            self.pixel_scale = self.native_pixel_scale / self.resampling_factor # the actual resolution
            self.resampling_factor = self.resampling_factor
        elif self.pixel_scale:
            self.pixel_scale = self.pixel_scale
            self.resampling_factor = self.native_pixel_scale/self.pixel_scale
        else:
            self.pixel_scale = self.native_pixel_scale
            self.resampling_factor = 1.0

        self.arcsec_per_proper_kpc = cosmo.arcsec_per_kpc_proper(z).value

        self.pixel_scale_kpc = self.pixel_scale/self.arcsec_per_proper_kpc

        self.ndim = int(self.target_width_arcsec/self.pixel_scale)

        self.width_arcsec = self.ndim * self.pixel_scale

        self.width = self.width_arcsec/self.arcsec_per_proper_kpc # width in kpc



        self.resolution = self.pixel_scale_kpc/self.super_sampling


        self.ndim_super = self.ndim * self.super_sampling

        # --- add sub-pixel offsets

        self.xoffset = xoffset_pix * self.pixel_scale_kpc
        self.yoffset = yoffset_pix * self.pixel_scale_kpc



        if verbose:
            print('*'*5, 'OBSERVED')
            print('z: {0}'.format(z))
            print('"/kpc: {0:.2f}'.format(self.arcsec_per_proper_kpc))
            print('-'*10)

            print('target width/": {0:.2f}'.format(self.target_width_arcsec))
            print('width/": {0:.2f}'.format(self.width_arcsec))
            print('width/kpc: {0:.2f}'.format(self.width))
            print('-'*10)

            print('native pixel scale/": {0:.2f}'.format(self.native_pixel_scale))
            print('pixel scale/": {0:.2f}'.format(self.pixel_scale))
            print('pixel scale/kpc: {0:.2f}'.format(self.pixel_scale_kpc))
            print('super sampling: {0:.2f}'.format(self.super_sampling))
            print('resolution/kpc: {0:.2f}'.format(self.resolution))
            print('ndim: {0:.2f}'.format(self.ndim))
            print('-'*10)


    def particle(self, X, Y, L):

        X -= np.median(X)
        Y -= np.median(Y)

        X += self.xoffset
        Y += self.yoffset


        imgs = empty()

        imgs.super = core(X, Y, L, resolution = self.resolution, ndim = self.ndim_super, smoothing = self.smoothing, verbose = self.verbose)
        imgs.super.no_PSF = imgs.super.data


        # --- apply PSF to super

        xx = yy = np.linspace(-(self.width_arcsec/self.native_pixel_scale/2.), (self.width_arcsec/self.native_pixel_scale/2.), self.ndim_super)

        psf = self.PSF.f(xx, yy)

        imgs.super.data = convolve_fft(imgs.super.no_PSF, psf)

        # --- resample back pixel scale


        imgs.img = empty()
        imgs.img.pixel_scale = self.pixel_scale
        imgs.img.pixel_scale_kpc = self.pixel_scale_kpc
        imgs.img.no_PSF = rebin(imgs.super.no_PSF, (self.ndim, self.ndim))
        imgs.img.data = rebin(imgs.super.data, (self.ndim, self.ndim))

        return imgs





    def Sersic(self, L, p):


        imgs = empty()


        g = np.linspace(-self.width/2., self.width/2.,  self.ndim_super) # in kpc

        xx, yy = np.meshgrid(g, g)

        mod = Sersic2D(amplitude = 1, r_eff = p['r_eff'], n = p['n'], x_0 = self.xoffset, y_0 = self.yoffset, ellip = p['ellip'], theta = p['theta'])

        sersic = mod(xx, yy)
        sersic /= np.sum(sersic)

        imgs.super = empty()
        imgs.super.pixel_scale = self.pixel_scale/self.super_sampling
        imgs.super.pixel_scale_kpc = self.pixel_scale_kpc/self.super_sampling
        imgs.super.no_PSF = L * sersic

        # --- apply PSF to super

        xx = yy = np.linspace(-(self.width_arcsec/self.pixel_scale/2.), (self.width_arcsec/self.pixel_scale/2.), self.ndim_super)

        psf = self.PSF.f(xx, yy)

        imgs.super.data = convolve_fft(imgs.super.no_PSF, psf)


        # --- resample back pixel scale


        imgs.img = empty()
        imgs.img.pixel_scale = self.pixel_scale
        imgs.img.pixel_scale_kpc = self.pixel_scale_kpc
        imgs.img.no_PSF = rebin(imgs.super.no_PSF, (self.ndim, self.ndim))
        imgs.img.data = rebin(imgs.super.data, (self.ndim, self.ndim))

        return imgs





def particle(X, Y, L, filters, cosmo, z, target_width_arcsec, resampling_factor=False, pixel_scale=False, smoothing = False, PSFs = False, super_sampling = 10, verbose = False, offsets = False):

    if offsets:
        xoffset_pix_base = np.random.random() - 0.5 # offset in pixels
        yoffset_pix_base = np.random.random() - 0.5 # offset in pixels
    else:
        xoffset_pix_base = yoffset_pix_base = 0.0

    # --- determine coarsest pixels

    max_pixel_scale = np.max([flare.observatories.filter_info[filter]['pixel_scale'] for filter in filters])

    IMGs = {}

    for filter in filters:

        xoffset_pix = xoffset_pix_base * (max_pixel_scale/flare.observatories.filter_info[filter]['pixel_scale'])
        yoffset_pix = yoffset_pix_base * (max_pixel_scale/flare.observatories.filter_info[filter]['pixel_scale'])

        imgs = observed(filter, cosmo, z, target_width_arcsec, resampling_factor = resampling_factor, pixel_scale = pixel_scale, smoothing = smoothing, PSF = PSFs[filter], super_sampling = super_sampling, verbose = verbose, xoffset_pix = xoffset_pix, yoffset_pix = xoffset_pix).particle(X, Y, L[filter])

        IMGs[filter] = imgs.img

    return IMGs



def Sersic(L, p, filters, cosmo, z, target_width_arcsec, resampling_factor=False, pixel_scale=False, smoothing = False, PSFs = False, super_sampling = 10, verbose = False, offsets = False):

    if offsets:
        xoffset_pix_base = np.random.random() - 0.5 # offset in pixels
        yoffset_pix_base = np.random.random() - 0.5 # offset in pixels
    else:
        xoffset_pix_base = yoffset_pix_base = 0.0

    # --- determine coarsest pixels

    max_pixel_scale = np.max([flare.filters.pixel_scale[filter] for filter in filters])

    IMGs = {}

    for filter in filters:

        xoffset_pix = xoffset_pix_base * (max_pixel_scale/flare.filters.pixel_scale[filter])
        yoffset_pix = yoffset_pix_base * (max_pixel_scale/flare.filters.pixel_scale[filter])

        imgs = observed(filter, cosmo, z, target_width_arcsec, resampling_factor = resampling_factor, pixel_scale = pixel_scale, smoothing = smoothing, PSF = PSFs[filter], super_sampling = super_sampling, verbose = verbose, xoffset_pix = xoffset_pix, yoffset_pix = xoffset_pix).Sersic(L[filter], p)

        IMGs[filter] = imgs.img

    return IMGs






def point(flux, filter, target_width_arcsec, resampling_factor = False, pixel_scale = False, PSF = False, verbose = False, super_sampling = 5, xoffset_pix = 0.0, yoffset_pix = 0.0):


    native_pixel_scale = flare.filters.pixel_scale[filter]

    if resampling_factor:
        pixel_scale = native_pixel_scale / resampling_factor # the actual resolution
        resampling_factor = resampling_factor
    elif pixel_scale:
        pixel_scale = pixel_scale
        resampling_factor = native_pixel_scale/pixel_scale
    else:
        pixel_scale = native_pixel_scale
        resampling_factor = 1.0

    # ndim = int(np.ceil(target_width_arcsec/pixel_scale) // 2 * 2 + 1) # forces odd number?

    ndim = int(target_width_arcsec/pixel_scale)

    ndim_super = ndim * super_sampling

    width_arcsec = ndim * pixel_scale

    xoffset_arcsec = xoffset_pix * pixel_scale
    yoffset_arcsec = yoffset_pix * pixel_scale


    if verbose:
        print('*'*5, 'POINT')
        print('-'*10)

        print('target width/": {0:.2f}'.format(target_width_arcsec))
        print('width/": {0:.2f}'.format(width_arcsec))
        print('-'*10)
        print('native pixel scale/": {0:.2f}'.format(native_pixel_scale))
        print('pixel scale/": {0:.2f}'.format(pixel_scale))
        print('super sampling: {0:.2f}'.format(super_sampling))
        print('ndim: {0:.2f}'.format(ndim))
        print('-'*10)



    imgs = empty()

    imgs.super = empty()
    imgs.super.ndim = ndim_super
    imgs.super.pixel_scale = pixel_scale/super_sampling

    imgs.super.hist = np.zeros((ndim_super, ndim_super))
    imgs.super.simple = np.zeros((ndim_super, ndim_super))

    g = np.linspace(-width_arcsec/2.,width_arcsec/2., ndim_super)
    i, j = (np.abs(g - xoffset_arcsec)).argmin(), (np.abs(g - yoffset_arcsec)).argmin()
    imgs.super.hist[j,i] += 1
    imgs.super.simple[j,i] += flux

    # --- apply PSF to super

    xx = yy = np.linspace(-(width_arcsec/native_pixel_scale)/2, width_arcsec/native_pixel_scale/2, ndim_super) # native pixel units

    psf = PSF.f(xx, yy)

    psf /= np.sum(psf)

    # --- renormalise PSF

    if verbose: print('sum(PSF width): {0:.2f}'.format(PSF.width))

    d = int(PSF.ndim*width_arcsec/PSF.width/2)
    c = PSF.ndim // 2
    psf_data = np.sum(PSF.data[c-d:c+d+1, c-d:c+d+1])/np.sum(PSF.data)
    if verbose: print('sum(psf_data): {0:.2f}'.format(psf_data))


    if verbose: print('sum(PSF): {0:.2f}'.format(np.sum(psf)))

    imgs.super.data = convolve_fft(imgs.super.simple, psf)*psf_data

    if verbose: print('sum(super.simple): {0:.2f}'.format(np.sum(imgs.super.simple)))
    if verbose: print('sum(super.data): {0:.2f}'.format(np.sum(imgs.super.data)))

    # --- resample back pixel scale

    imgs.img = empty()
    imgs.img.ndim = ndim
    imgs.img.pixel_scale = pixel_scale

    imgs.img.no_PSF = rebin(imgs.super.simple, (ndim, ndim))
    imgs.img.data = rebin(imgs.super.data, (ndim, ndim))

    imgs.img.data /= np.sum(imgs.img.data)
    imgs.img.data *= flux

    # imgs.img.data = convolve(imgs.img.data, PSF.charge_diffusion_kernel)

    return imgs




def points(fluxes, filters, width_arcsec, resampling_factor = False, pixel_scale = False, PSFs = False, verbose = False):

    xoffset_pix = np.random.random() - 0.5 # offset in pixels
    yoffset_pix = np.random.random() - 0.5 # offset in pixels

    IMGs = {}

    for filter in filters:

        imgs = point(fluxes[filter], filter, width_arcsec, resampling_factor = resampling_factor, pixel_scale = pixel_scale, PSF = PSFs[filter], verbose = verbose, xoffset_pix = xoffset_pix, yoffset_pix = yoffset_pix)
        IMGs[filter] = imgs.img

    return IMGs
