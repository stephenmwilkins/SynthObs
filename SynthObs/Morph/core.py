

from scipy.spatial import cKDTree
import numpy as np

import matplotlib.pyplot as plt

















class physical_image():

    def __init__(self, X, Y, L, resolution = 0.1, Ndim = 100, smoothed = True, show = False):

        X -= np.median(X)
        Y -= np.median(Y)

        self.smoothed = smoothed

        self.Ndim = Ndim
        self.resolution = resolution
        self.width = Ndim * resolution 

        range = [np.max(X) - np.min(X), np.max(Y) - np.min(Y)]
        print(range) 

        if any(x>Ndim*resolution for x in range): print('Warning particles will extend beyond image limits')

        if not self.smoothed: g = np.linspace(-self.width/2.,self.width/2.,Ndim)

        if self.smoothed: Gx, Gy = np.meshgrid(np.linspace(-self.width/2.,self.width/2.,Ndim), np.linspace(-self.width/2.,self.width/2.,Ndim))

        self.img = np.zeros((self.Ndim, self.Ndim))

#         if self.smoothed:
#          
#             for x,y,l, in zip(X, Y, L):
# 
#                 R = np.sqrt((X-x)**2 + (Y-y)**2)  
#               
#                 if len(self.X)>7:
#                     r = np.max([sorted(R)[7], 0.1])
#                 else:
#                     r = sorted(R)[-1]
#     
#                 gauss = np.exp(-(((Gx - x)**2 + (Gy - y)**2)/ ( 2.0 * r**2 ) ) )  
# 
#                 sgauss = np.sum(gauss)
# 
#                 if sgauss > 0: self.img += l*gauss/sgauss


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

     




class observed_image():

    def __init__(self, X, Y, fluxes, f, cosmo, redshift = 8, base_width = 10, resampling_factor = 1.0, smoothed = True, add_PSF = True):


        # --- X in kpc
        # --- Y in kpc
        # --- fluxes in nJy
        # --- filter (e.g. JWST.NIRCAM.F115W)
        # --- cosmology (astro.cosmology object)
        # --- redshift
        # --- base_width (arcsec) APPROXIMATE 
        # --- resampling_factor = factor by which to resample the PSF
        # --- smoothed (apply 7th nearest neighbour smoothing)
        # --- apply PSF (apply 7th nearest neighbour smoothing)

        X -= np.median(X)
        Y -= np.median(Y)

        X_arcsec = # DO THIS
        Y_arcsec = # DO THIS
        
        self.resampling_factor = resampling_factor
        
        self.base_pixel_scale = # the base instrument pixel scale before resampling 

        self.pixel_scale = self.base_pixel_scale / self.resampling_factor # the final image pixel scale


        self.smoothed = smoothed

        self.Ndim = int(base_width/self.pixel_scale)
        
        self.width

        self.Ndim = Ndim
        
        self.width = Ndim * resolution 

        range = [np.max(X) - np.min(X), np.max(Y) - np.min(Y)]
        print(range) 

        if any(x>Ndim*resolution for x in range): print('Warning particles will extend beyond image limits')

        if not self.smoothed: g = np.linspace(-self.width/2.,self.width/2.,Ndim)

        if self.smoothed: Gx, Gy = np.meshgrid(np.linspace(-self.width/2.,self.width/2.,Ndim), np.linspace(-self.width/2.,self.width/2.,Ndim))

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
        
        