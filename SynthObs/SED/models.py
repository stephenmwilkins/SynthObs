

import numpy as np
import pickle

from FLARE.SED import core


class define_model():

    def __init__(self, grid, path_to_SPS_grid = '', dust = False):
    
        self.grid = pickle.load(open(path_to_SPS_grid + grid + '/nebular.p','rb'), encoding='latin1')

        self.lam = self.grid['lam']

        self.dust = dust
        


class generate_SED():

    
    def __init__(self, model, Masses, Ages, Metallicities, MetSurfaceDensities, include_intrinsic = True, IGM = False):


        self.model = model
        self.lam = self.model.grid['lam'] # convenience

        self.stellar = core.sed(self.lam)
        self.nebular = core.sed(self.lam)
        self.total = core.sed(self.lam)
    
        if include_intrinsic:
    
            self.intrinsic_stellar = core.sed(self.lam)
            self.intrinsic_nebular = core.sed(self.lam)
            self.intrinsic_total = core.sed(self.lam)
    

        for Mass, Age, Metallicity, MetalSurfaceDensity in zip(Masses, Ages, Metallicities, MetSurfaceDensities):


            log10age = np.log10(Age) + 6. # log10(age/yr)
            log10Z = np.log10(Metallicity) # log10(Z)
        

            # --- determine dust attenuation

            if self.model.dust:
    
                tau_V = (10**self.model.dust['A']) * MetalSurfaceDensity                     

                tau = tau_V * (self.lam/5500.)**self.model.dust['slope']
    
                T = np.exp(-tau)
    
            else:
    
                T = 1.0
    
    
            # --- determine closest SED grid point 

            ia = (np.abs(self.model.grid['log10age'] - log10age)).argmin()
            iZ = (np.abs(self.model.grid['log10Z'] - log10Z)).argmin()
 
            self.stellar.lnu += Mass * T * self.model.grid['stellar'][ia, iZ] # erg/s/Hz
            self.nebular.lnu += Mass * T * self.model.grid['nebular'][ia, iZ] # erg/s/Hz

            if include_intrinsic:

                self.intrinsic_stellar.lnu += Mass * self.model.grid['stellar'][ia, iZ] # erg/s/Hz
                self.intrinsic_nebular.lnu += Mass * self.model.grid['nebular'][ia, iZ] # erg/s/Hz



        self.total.lnu = self.stellar.lnu + self.nebular.lnu # erg/s/Hz
        
        if include_intrinsic: self.intrinsic_total.lnu = self.intrinsic_stellar.lnu + self.intrinsic_nebular.lnu # erg/s/Hz

        

        
            
        
        


        
 