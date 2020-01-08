

import numpy as np
import pickle


class empty(): pass

from ..core import *

from FLARE.SED import core
from FLARE.SED import IGM

# from scipy import ndimage # -- used for interpolation

class define_model():

    def __init__(self, grid, path_to_SPS_grid = FLARE_dir + '/data/SPS/nebular/2.0/Z_refQ_wdust/', dust = False):

        self.grid = pickle.load(open(path_to_SPS_grid + grid + '/nebular.p','rb'), encoding='latin1')

        self.lam = self.grid['lam']

        self.dust = dust

        # --- add backwards grid compatibility

        if 'stellar_incident' not in self.grid:
            print('WARNING: you are using old grids!')
            self.grid['stellar_transmitted'] = self.grid['stellar']
            self.grid['stellar_incident'] = self.grid['stellar']


    def create_Lnu_grid(self, F):

        self.Lnu = {}

        for f in F['filters']:

            # self.Lnu[f] = np.trapz(np.multiply(self.grid['nebular'] + self.grid['stellar'], F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam)

            self.Lnu[f] = empty()

            self.Lnu[f].stellar_incident = np.trapz(np.multiply(self.grid['stellar_incident'], F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam)
            self.Lnu[f].stellar_transmitted = np.trapz(np.multiply(self.grid['stellar_transmitted'], F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam)
            self.Lnu[f].nebular = np.trapz(np.multiply(self.grid['nebular'], F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam)



    def create_Fnu_grid(self, F, z, cosmo):

        self.Fnu = {}

        self.Fnu['z'] = z

        luminosity_distance = cosmo.luminosity_distance(z).to('cm').value

        for f in F['filters']:

            self.Fnu[f] = empty()

            self.Fnu[f].stellar_incident = 1E23 * 1E9 * np.trapz(np.multiply((self.grid['stellar_incident'])*IGM.madau(F[f].lam, z), F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam) * (1.+z) / (4. * np.pi * luminosity_distance**2)
            self.Fnu[f].stellar_transmitted = 1E23 * 1E9 * np.trapz(np.multiply((self.grid['stellar_transmitted'])*IGM.madau(F[f].lam, z), F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam) * (1.+z) / (4. * np.pi * luminosity_distance**2)
            self.Fnu[f].nebular = 1E23 * 1E9 * np.trapz(np.multiply((self.grid['nebular'])*IGM.madau(F[f].lam, z), F[f].T), x = F[f].lam, axis = 2)/np.trapz(F[f].T, x = F[f].lam) * (1.+z) / (4. * np.pi * luminosity_distance**2)






def generate_Lnu(model, Masses, Ages, Metallicities, tauVs, F, fesc = 0.0, log10t_BC = 7.):

    L = {f: 0.0 for f in F['filters']}

    for f in F['filters']:

        L[f] = np.sum(generate_Lnu_array(model, Masses, Ages, Metallicities, tauVs, F, f, fesc = fesc, log10t_BC = log10t_BC))

    return L


def generate_Lnu_array(model, Masses, Ages, Metallicities, tauVs, F, f, fesc = 0.0, log10t_BC = 7.):

    # --- determine dust attenuation

    if model.dust: tau_f = (F[f].pivwv()/5500.)**model.dust['slope']

    l = np.zeros(Masses.shape)

    for i, Mass, Age, Metallicity, tauV in zip(np.arange(Masses.shape[0]), Masses, Ages, Metallicities, tauVs):

        log10age = np.log10(Age) + 6. # log10(age/yr)
        log10Z = np.log10(Metallicity) # log10(Z)

        # --- determine closest SED grid point

        ia = (np.abs(model.grid['log10age'] - log10age)).argmin()
        iZ = (np.abs(model.grid['log10Z'] - log10Z)).argmin()

        if model.dust:

            T = np.exp(-(tauV * tau_f))

        else:

            T = 1.0

        # --- apply birth cloud model

        if log10age < log10t_BC:
            l[i] = Mass * T * (fesc*model.Lnu[f].stellar_incident[ia, iZ] + (1.-fesc)*(model.Lnu[f].stellar_transmitted[ia, iZ] + model.Lnu[f].nebular[ia, iZ]))
        else:
            l[i] = Mass * T * model.Lnu[f].stellar_incident[ia, iZ]


    return l



def generate_Fnu(model, Masses, Ages, Metallicities, tauVs, F, fesc = 0.0, log10t_BC = 7.):

    Fnu = {f: 0.0 for f in F['filters']}

    for f in F['filters']:

        Fnu[f] = np.sum(generate_Fnu_array(model, Masses, Ages, Metallicities, tauVs, F, f, fesc = fesc, log10t_BC = log10t_BC))

    return Fnu


def generate_Fnu_array(model, Masses, Ages, Metallicities, tauVs, F, f, fesc = 0.0, log10t_BC = 7.):

    # --- determine dust attenuation

    if model.dust: tau_f = (F[f].pivwv()/(5500.*(1.+model.Fnu['z'])))**model.dust['slope']

    l = np.zeros(Masses.shape)

    for i, Mass, Age, Metallicity, tauV in zip(np.arange(Masses.shape[0]), Masses, Ages, Metallicities, tauVs):

        log10age = np.log10(Age) + 6. # log10(age/yr)
        log10Z = np.log10(Metallicity) # log10(Z)

                # --- determine closest SED grid point

        ia = (np.abs(model.grid['log10age'] - log10age)).argmin()
        iZ = (np.abs(model.grid['log10Z'] - log10Z)).argmin()

        if model.dust:
            T = np.exp(-(tauV * tau_f))
        else:
            T = 1.0

        # --- apply birth cloud model

        if log10age < log10t_BC:
            l[i] = Mass * T * (fesc*model.Fnu[f].stellar_incident[ia, iZ] + (1.-fesc)*(model.Fnu[f].stellar_transmitted[ia, iZ] + model.Fnu[f].nebular[ia, iZ]))
        else:
            l[i] = Mass * T * model.Fnu[f].stellar_incident[ia, iZ]


    return l



class generate_SED():


    def __init__(self, model, Masses, Ages, Metallicities, tauVs, IGM = False, fesc = 0.0, log10t_BC = 7.):


        self.model = model
        self.lam = self.model.grid['lam'] # convenience

        self.intrinsic = core.sed(self.lam, description = 'intrinsic (stellar) SED') # includes no reprocessing by gas or dust
        self.no_BC = core.sed(self.lam, description = 'SED with no birth cloud but still including ISM dust (same as setting fesc=1.0)') # only includes dust in ISM, no BC (thus no nebular emission)
        self.no_ISM = core.sed(self.lam, description = 'SED with birth cloud effects but no ISM dust') # includes gas/dust reprocessing in BC, only particles with age < t_BC contribute
        self.total = core.sed(self.lam, description = 'SED including both birth cloud and ISM dust')


        for Mass, Age, Metallicity, tauV in zip(Masses, Ages, Metallicities, tauVs):

            log10age = np.log10(Age) + 6. # log10(age/yr)
            log10Z = np.log10(Metallicity) # log10(Z)

            # --- determine closest SED grid point

            ia = (np.abs(self.model.grid['log10age'] - log10age)).argmin()
            iZ = (np.abs(self.model.grid['log10Z'] - log10Z)).argmin()

            self.intrinsic.lnu += Mass * self.model.grid['stellar_incident'][ia, iZ]


            # --- determine ISM dust attenuation

            if self.model.dust:
                tau = tauV * (self.lam/5500.)**self.model.dust['slope']
                T = np.exp(-tau)
            else:
                T = 1.0


            self.no_BC.lnu += T * Mass * self.model.grid['stellar_incident'][ia, iZ]


            if log10age < log10t_BC:

                self.no_ISM.lnu += Mass * (fesc*self.model.grid['stellar_incident'][ia, iZ] + (1.-fesc)*(self.model.grid['stellar_transmitted'][ia, iZ] + self.model.grid['nebular'][ia, iZ]))

                self.total.lnu += Mass * T * (fesc*self.model.grid['stellar_incident'][ia, iZ] + (1.-fesc)*(self.model.grid['stellar_transmitted'][ia, iZ] + self.model.grid['nebular'][ia, iZ]))

            else:

                self.no_ISM.lnu += Mass * self.model.grid['stellar_incident'][ia, iZ]

                self.total.lnu += Mass * T * self.model.grid['stellar_incident'][ia, iZ]
























class EmissionLines():

    def __init__(self, SPSIMF, dust = False, verbose = True):

        self.SPSIMF = SPSIMF

        self.grid = pickle.load(open(FLARE_dir + f'/data/SPS/nebular/2.0/Z_refQ/{SPSIMF}/lines.p','rb'), encoding='latin1')  # --- open grid

        self.lines = self.grid['lines']

        self.lam = {l: self.grid[l]['lam'] for l in self.lines}

        self.lams = np.array([self.lam[l] for l in self.lines])

        if verbose:
            print('Available lines:')
            for lam,line in sorted(zip(self.lams,self.lines)): print(f'{line}')


        self.dust = dust

        self.units = {'luminosity': 'erg/s', 'nebular_continuum': 'erg/s/Hz', 'stellar_continuum': 'erg/s/Hz', 'total_continuum': 'erg/s/Hz', 'EW': 'AA'}





    def get_line_luminosity(self, line, Masses, Ages, Metallicities, tauVs = False, fesc = False, verbose = False):


        if type(line) is not list: line = [line]

        if type(tauVs) is not np.ndarray:
            tauVs = np.zeros(Masses.shape)
            if verbose: 'WARNING: no optical depths provided, quantities will be intrinsic!'

        if not self.dust:
            if verbose: 'WARNING: no dust model specified, quantities will be intrinsic!'


        lam = np.mean([self.grid[l]['lam'] for l in line])

        if verbose:
            print(f'----- {line}')
            print(f'line wavelength/\AA: {lam}')


        l_types = ['luminosity', 'nebular_continuum', 'stellar_continuum', 'total_continuum']

        o = {l_type: 0.0 for l_type in l_types} # output dictionary
#         o['lam'] = lam
#         o['line'] = line



        for Mass, Age, Metallicity, tauV in zip(Masses, Ages, Metallicities, tauVs):

            log10age = np.log10(Age) + 6. # log10(age/yr)
            log10Z = np.log10(Metallicity) # log10(Z)

            # --- determine dust attenuation

            if self.dust:

                tau = tauV * (lam/5500.)**self.dust['slope']

                T = np.exp(-tau)

            else:

                T = 1.0


            # --- determine closest SED grid point

            ia = (np.abs(self.grid['log10age'] - log10age)).argmin()
            iZ = (np.abs(self.grid['log10Z'] - log10Z)).argmin()

            for l in line:
                for l_type in l_types:
                    if l_type == 'luminosity':
                        o[l_type] += Mass * T * 10**self.grid[l][l_type][ia, iZ] # erg/s
                    else:
                        o[l_type] += Mass * T * self.grid[l][l_type][ia, iZ] # erg/s

        if fesc:
            for l_type in l_types: o[l_type] *= 1-fesc


        total_continuum = (o['total_continuum']/float(len(line)))*(3E8)/((lam/float(len(line)))**2*1E-10)

        o['EW'] = o['luminosity']/total_continuum

        if verbose:
            for k,v in o.items(): print(f'log10({k}/{self.units[k]}): {np.log10(v):.2f}')

        return o
