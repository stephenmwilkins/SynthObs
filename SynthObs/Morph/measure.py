


import numpy as np

from photutils import detect_sources
from photutils import source_properties
from photutils import CircularAperture
from photutils import aperture_photometry

import matplotlib.pyplot as plt

import scipy.ndimage
from scipy.optimize import minimize

from scipy.spatial import cKDTree






def simple(X, Y, L):

    star_pos = np.column_stack([X, Y])

    # Build a tree out of star particles
    tree = cKDTree(star_pos, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)

    # Query tree for all particle separations from the centre (mean particle position)
    nndists, nninds = tree.query(np.mean(star_pos, axis=0), k=len(star_pos), n_jobs=-1)

    # Sort luminosities by distance to centre using indices returned by
    srt_lums = L[nninds]

    # Compute max flux
    tot_lum = np.sum(srt_lums)

    # Compute half the total flux
    half_lum = tot_lum / 2

    # Loop through stars until the flux is greater than half
    # NOTE: due to granular nature of star luminosity the intrinsic half light radius carries a lower error equal
    # to the distance between the previous star and the one with flux greater than half the total
    ind = 0
    flux = 0
    pre_dist = 0
    hl_r = -1
    hl_err = -1
    while flux < half_lum:

       # Add this stars flux
       flux += srt_lums[ind]

       # Assign the distance
       hl_r = nndists[ind]

       # Compute the error on this radius
       hl_err = hl_r - pre_dist

       # Assign this distance to the previous distance
       pre_dist = hl_r

       # Increment the index counter
       ind += 1

    return hl_r, hl_err












class intrinsic():


    def __init__(self, img):
     
        self.img = img
        
        self.profile = False
        
        self.total = np.sum(self.img.img)

    def detect_sources(self, threshold = False, npixels = 10):

        if not threshold:
    
            # threshold = np.min(self.img[self.img>0])
        
            threshold = np.sum(self.img.img)/1000.
        
        self.segm = detect_sources(self.img.img, threshold, npixels = npixels)
    
        self.cat = source_properties(self.img.img, self.segm)
    
#         for i, o in enumerate(self.cat):
#             print(i, o.centroid, o.source_sum/np.sum(self.img.img))
            
                     
    def measure_profile(self, objid = 0, dr = 0.1, show = False): # --- measure the profile
    
        if not self.cat: self.detect_sources()
    
        radii = np.arange(dr, self.img.width/self.img.resolution, dr) # in pixels
        positions = [self.cat[objid].centroid] # centre
        apertures = [CircularAperture(positions, r=r) for r in radii] #r in pixels
        phot_table = aperture_photometry(self.img.img, apertures) 
    
        self.profile = {'r_pix': radii, 'r_kpc': radii*self.img.resolution, 'I': np.array([float(phot_table[0][j+3]) for j in range(len(radii))])}
        
        if show:
        
            plt.plot(self.profile['r_kpc'], self.profile['I']) 
            plt.show()  
    
    
    
    def rcurve(self):
    
        if not self.profile: self.measure_profile()
        
        return np.interp(0.5, self.profile['I']/self.total, self.profile['r_kpc'])
           
    
    def rpix(self): # --- measure r_e sing the pixel method
    
        sortpix = np.array(sorted(self.img.img.flatten())[::-1])
        cumsum = np.cumsum(sortpix)/sum(sortpix)
        npix = len(cumsum[cumsum<0.5])
        area = npix*(self.img.resolution)**2
        
        return np.sqrt(area/np.pi)


    def fit_sersic(self, n = False): # --- fit by a Sersic function
    
        if not self.profile: self.measure_profile()
 
        r = self.profile['r_kpc']
        dr = r[1] - r[0]
        I = self.profile['I']
    
        SB = (I[1:]-I[:-1])/(np.pi*(r[1:]**2 - r[:-1]**2))         
        nr = r[:-1]+dr/2.
    
        guess = (self.rpix(), np.interp(self.rpix(), nr, SB), 1.5)
        
        if n: # --- if index given do a forced fit

            guess = (self.rpix(), np.interp(self.rpix(), nr, SB))
            sersic = lambda r, r_e, I_e: I_e * np.exp(-(1.9992*n - 0.3271)*((r/r_e)**(1./n) - 1.0))
            popt, pcov = scipy.optimize.curve_fit(sersic, nr, SB, guess)
            
        else: # --- if index is NOT given do a free fit
        
            guess = (self.rpix(), np.interp(self.rpix(), nr, SB), 1.5)
            sersic = lambda r, r_e, I_e, n: I_e * np.exp(-(1.9992*n - 0.3271)*((r/r_e)**(1./n) - 1.0))
            popt, pcov = scipy.optimize.curve_fit(sersic, nr, SB, guess)
        
        return popt

        
    def rsersic(self, n = False): # --- fit by a Sersic function
    
        return self.fit_sersic(n)[0]
 

#     def rpetrosian(self): # ------ determine Petrosian radius
#         
#         eta = surface_brightness/np.interp(nradii, radii, enclosed_average_surface_brightness)
#         
#         r_p_pix = np.interp(0.2, eta[::-1], nradii[::-1])
#         r_p = r_p_pix*res


    def r_e(self):
    
        return {'pixel': self.rpix(), 'curve': self.rcurve(), 'sersic1': self.rsersic(n=1), 'sersic': self.rsersic()} 
        
        
        
#     def A(self): # --- assymmetry
#     
#         rotated = scipy.ndimage.interpolation.rotate(self.img.img, 180.)
#         return np.sum(np.fabs(self.img.img-rotated))/np.sum(self.img.img)
        
#     def M20(self):
#     
#         m = np.linspace(-self.img.Ndim/2.,self.img.Ndim/2.,self.img.Ndim)
#     
#         Mx, My = np.meshgrid(m, m)
#     
#         R = Mx**2+My**2
#         
#         flux = self.img.img.flatten()
#         
#         Mtot = np.sum(R.flatten()*flux)
#         
#         r = R.flatten()
#         
#         inds = np.argsort(flux)[::-1] # sorted from biggest to smallest
#         
#         sortflux = flux[inds]
#         sortr = r[inds]
#         
#         cumsum = np.cumsum(sortflux)/sum(sortflux)
#         
#         maxi = len(cumsum[cumsum<0.2])
#         
#         return np.log10(np.sum(sortflux[:maxi]*sortr[:maxi])/Mtot)


#     def C(self):
#     
#         Concentration = 5*np.log10(np.interp(0.8, C, radii)/np.interp(0.2, C, radii))
#     
#         C_pet = C/np.interp(1.5*r_p_pix, radii, C)
#         
#         Concentration_pet = 5*np.log10(np.interp(0.8, C_pet, radii)/np.interp(0.2, C_pet, radii))
        
    
    
#     def Gini(self):
#         
#         
#         mu = np.interp(r_p_pix, nradii, surface_brightness)
#         
#         sortpix = np.array(sorted(img.flatten()))
#            
#         ginipix = sortpix[sortpix>mu]
#         
#         n = len(ginipix)
#   
#         i = np.arange(1,n+1,1)
#         
#         Gini = (1./(np.mean(ginipix)*n*(n-1)))*np.sum((2*i-n-1)*ginipix)
        
        
        
        
        