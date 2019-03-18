

import numpy as np
import pickle


class empty(): pass

from ..core import * 

from FLARE.SED import core
from FLARE.SED import IGM

# from scipy import ndimage # -- used for interpolation

class define_model():

    def __init__(self, grid, path_to_SPS_grid = FLARE_dir + 'data/SPS/nebular/1.0/Z/', dust = False):
    
        self.grid = pickle.load(open(path_to_SPS_grid + grid + '/nebular.p','rb'), encoding='latin1')

        self.lam = self.grid['lam']

        self.dust = dust
        

    def create_Lnu_grid(self, F):
    
        self.Lnu = {}
    
        for f in F['filters']:
        
            # self.Lnu[f] = np.trapz(np.multiply(self.grid['nebular'] + self.grid['stellar'], F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam)
        
            self.Lnu[f] = empty()
            
            self.Lnu[f].stellar = np.trapz(np.multiply(self.grid['stellar'], F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam)
            self.Lnu[f].nebular = np.trapz(np.multiply(self.grid['nebular'], F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam)
        
            
        
    def create_Fnu_grid(self, F, z, cosmo):
    
        self.Fnu = {}
    
        self.Fnu['z'] = z
    
        luminosity_distance = cosmo.luminosity_distance(z).to('cm').value
    
        for f in F['filters']:
            
            self.Fnu[f] = empty()
            
            self.Fnu[f].stellar = 1E23 * 1E9 * np.trapz(np.multiply((self.grid['stellar'])*IGM.madau(F[f].lam, z), F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam) * (1.+z) / (4. * np.pi * luminosity_distance**2)
            self.Fnu[f].nebular = 1E23 * 1E9 * np.trapz(np.multiply((self.grid['nebular'])*IGM.madau(F[f].lam, z), F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam) * (1.+z) / (4. * np.pi * luminosity_distance**2)
        
            




def generate_Lnu(model, Masses, Ages, Metallicities, MetSurfaceDensities, F, fesc = 0.0):

    L = {f: 0.0 for f in F['filters']}

    for f in F['filters']:
    
        L[f] = np.sum(generate_Lnu_array(model, Masses, Ages, Metallicities, MetSurfaceDensities, F, f, fesc = fesc))

    return L


def generate_Lnu_array(model, Masses, Ages, Metallicities, MetSurfaceDensities, F, f, fesc = 0.0):

    # --- determine dust attenuation

    if model.dust: tau_f = (F[f].pivwv()/5500.)**model.dust['slope']

    l = np.zeros(Masses.shape)

    for i, Mass, Age, Metallicity, MetalSurfaceDensity in zip(np.arange(Masses.shape[0]), Masses, Ages, Metallicities, MetSurfaceDensities):

        log10age = np.log10(Age) + 6. # log10(age/yr)
        log10Z = np.log10(Metallicity) # log10(Z)

        ia = (np.abs(model.grid['log10age'] - log10age)).argmin()
        iZ = (np.abs(model.grid['log10Z'] - log10Z)).argmin()

        if model.dust:

            tau_V = (10**model.dust['A']) * MetalSurfaceDensity  
            
            T = np.exp(-(tau_V * tau_f)) 

        else:

            T = 1.0

        # --- determine closest SED grid point 

        l[i] = Mass * T * (model.Lnu[f].stellar[ia, iZ] + (1.-fesc)*model.Lnu[f].nebular[ia, iZ]) # erg/s/Hz

        # --- use interpolation [this appears to make a difference at the low-% level, far below other systematic uncertainties]
    
        # p = {'log10age': log10age, 'log10Z': log10Z}
        # params = [[np.interp(p[parameter], model.grid[parameter], range(len(model.grid[parameter])))] for parameter in ['log10age','log10Z']] # used in interpolation
        # L[f] +=  Mass * T * ndimage.map_coordinates(model.L[f], params, order=1)[0]


    return l



def generate_Fnu(model, Masses, Ages, Metallicities, MetSurfaceDensities, F, fesc = 0.0):

    Fnu = {f: 0.0 for f in F['filters']}

    for f in F['filters']:
    
        Fnu[f] = np.sum(generate_Fnu_array(model, Masses, Ages, Metallicities, MetSurfaceDensities, F, f, fesc = fesc))

    return Fnu


def generate_Fnu_array(model, Masses, Ages, Metallicities, MetSurfaceDensities, F, f, fesc = 0.0):

    # --- determine dust attenuation

    if model.dust: tau_f = (F[f].pivwv()/(5500.*(1.+model.Fnu['z'])))**model.dust['slope']

    l = np.zeros(Masses.shape)

    for i, Mass, Age, Metallicity, MetalSurfaceDensity in zip(np.arange(Masses.shape[0]), Masses, Ages, Metallicities, MetSurfaceDensities):

        log10age = np.log10(Age) + 6. # log10(age/yr)
        log10Z = np.log10(Metallicity) # log10(Z)

        ia = (np.abs(model.grid['log10age'] - log10age)).argmin()
        iZ = (np.abs(model.grid['log10Z'] - log10Z)).argmin()

        if model.dust:

            tau_V = (10**model.dust['A']) * MetalSurfaceDensity  
            
            T = np.exp(-(tau_V * tau_f)) 

        else:

            T = 1.0

        # --- determine closest SED grid point 

        # l[i] = Mass * T * model.Fnu[f][ia, iZ] 
        
        l[i] = Mass * T * (model.Fnu[f].stellar[ia, iZ] + (1.-fesc)*model.Fnu[f].nebular[ia, iZ]) # erg/s/Hz

        # --- use interpolation [this appears to make a difference at the low-% level, far below other systematic uncertainties]
    
        # p = {'log10age': log10age, 'log10Z': log10Z}
        # params = [[np.interp(p[parameter], model.grid[parameter], range(len(model.grid[parameter])))] for parameter in ['log10age','log10Z']] # used in interpolation
        # L[f] +=  Mass * T * ndimage.map_coordinates(model.L[f], params, order=1)[0]


    return l





class generate_SED():

    
    def __init__(self, model, Masses, Ages, Metallicities, MetSurfaceDensities, include_intrinsic = True, IGM = False, fesc = 0.0):


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



        self.total.lnu = self.stellar.lnu + (1-fesc) * self.nebular.lnu # erg/s/Hz
        
        if include_intrinsic: self.intrinsic_total.lnu = self.intrinsic_stellar.lnu + self.intrinsic_nebular.lnu # erg/s/Hz

        

        
            
        
        


        
 