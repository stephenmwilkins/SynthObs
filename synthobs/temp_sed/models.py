

import numpy as np
import pickle


class empty(): pass

from ..core import *

from FLARE.SED import core
from FLARE.SED import IGM
from FLARE.SED import dust_curves

# from scipy import ndimage # -- used for interpolation

class define_model():

    def __init__(self, grid, path_to_SPS_grid = FLARE_dir + '/data/SPS/nebular/3.0/'):

        self.grid = pickle.load(open(path_to_SPS_grid + grid + '/nebular.p','rb'), encoding='latin1')

        self.lam = self.grid['lam']

        self.dust_ISM = False
        self.dust_BC = False


        if 'stellar_incident' not in self.grid:

            # --- this is now the canonical way of doing things
            self.grid['stellar_transmitted'] = self.grid['stellar']
            self.grid['stellar_transmitted'][:,:,self.lam<912] = 0.0
            self.grid['stellar_incident'] = self.grid['stellar']


    def create_Lnu_grid(self, F):

        self.Lnu = {}

        for f in F['filters']:

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






def generate_Lnu(model, Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC, F, fesc = 0.0, log10t_BC = 7.):

    L = {f: 0.0 for f in F['filters']}

    for f in F['filters']:

        L[f] = np.sum(generate_Lnu_array(model, Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC, F, f, fesc = fesc, log10t_BC = log10t_BC))

    return L


def generate_Lnu_array(model, Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC, F, f, fesc = 0.0, log10t_BC = 7.):

    # --- determine dust attenuation, this is somewhat problematic as it assumes the value of the dust curve in the middle of the band

    if model.dust_ISM:
        dust_model, dust_model_params = model.dust_ISM
        tau_ISM_lam = getattr(dust_curves, dust_model)(params = dust_model_params).tau(F[f].pivwv()) # optical depth relative to tau_V for wavelengths

    if model.dust_BC:
        dust_model, dust_model_params = model.dust_BC
        tau_BC_lam = getattr(dust_curves, dust_model)(params = dust_model_params).tau(F[f].pivwv()) # optical depth relative to tau_V for wavelengths


    l = np.zeros(Masses.shape)

    for i, Mass, Age, Metallicity, tauV_ISM, tauV_BC in zip(np.arange(Masses.shape[0]), Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC):

        log10age = np.log10(Age) + 6. # log10(age/yr)
        log10Z = np.log10(Metallicity) # log10(Z)

        # --- determine closest SED grid point

        ia = (np.abs(model.grid['log10age'] - log10age)).argmin()
        iZ = (np.abs(model.grid['log10Z'] - log10Z)).argmin()


        if model.dust_ISM:
            T_ISM = np.exp(-(tauV_ISM * tau_ISM_lam))
        else:
            T_ISM = 1.0

        # --- apply birth cloud model

        if log10age < log10t_BC:

            if model.dust_BC:
                T_BC = np.exp(-(tauV_BC * tau_BC_lam))
            else:
                T_BC = 1.0

            l[i] = Mass * (T_ISM*fesc*model.Lnu[f].stellar_incident[ia, iZ] + (T_ISM * T_BC)*(1.-fesc)*(model.Lnu[f].stellar_transmitted[ia, iZ] + model.Lnu[f].nebular[ia, iZ]))
        else:
            l[i] = Mass * T_ISM * model.Lnu[f].stellar_incident[ia, iZ]

    return l



def generate_Fnu(model, Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC, F, fesc = 0.0, log10t_BC = 7.):

    Fnu = {f: 0.0 for f in F['filters']}

    for f in F['filters']:

        Fnu[f] = np.sum(generate_Fnu_array(model, Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC, F, f, fesc = fesc, log10t_BC = log10t_BC))

    return Fnu


def generate_Fnu_array(model, Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC, F, f, fesc = 0.0, log10t_BC = 7.):

    # --- determine dust attenuation

    if model.dust_ISM:
        dust_model, dust_model_params = model.dust_ISM
        tau_ISM_lam = getattr(dust_curves, dust_model)(params = dust_model_params).tau(F[f].pivwv()/(1.+model.Fnu['z'])) # optical depth relative to tau_V for wavelengths

    if model.dust_BC:
        dust_model, dust_model_params = model.dust_BC
        tau_BC_lam = getattr(dust_curves, dust_model)(params = dust_model_params).tau(F[f].pivwv()/(1.+model.Fnu['z'])) # optical depth relative to tau_V for wavelengths


    l = np.zeros(Masses.shape)

    for i, Mass, Age, Metallicity, tauV_ISM, tauV_BC in zip(np.arange(Masses.shape[0]), Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC):

        log10age = np.log10(Age) + 6. # log10(age/yr)
        log10Z = np.log10(Metallicity) # log10(Z)

                # --- determine closest SED grid point

        ia = (np.abs(model.grid['log10age'] - log10age)).argmin()
        iZ = (np.abs(model.grid['log10Z'] - log10Z)).argmin()

        if model.dust_ISM:
            T_ISM = np.exp(-(tauV_ISM * tau_ISM_lam))
        else:
            T_ISM = 1.0

        # --- apply birth cloud model

        if log10age < log10t_BC:

            if model.dust_BC:
                T_BC = np.exp(-(tauV_BC * tau_BC_lam))
            else:
                T_BC = 1.0

            l[i] = Mass * (T_ISM*fesc*model.Fnu[f].stellar_incident[ia, iZ] + (T_ISM * T_BC)*(1.-fesc)*(model.Fnu[f].stellar_transmitted[ia, iZ] + model.Fnu[f].nebular[ia, iZ]))
        else:
            l[i] = Mass * T_ISM * model.Fnu[f].stellar_incident[ia, iZ]

    return l



class generate_SED():


    def __init__(self, model, Masses, Ages, Metallicities, tauVs_ISM = False, tauVs_BC = False, IGM = False, fesc = 0.0, log10t_BC = 7.):

        self.model = model
        self.lam = self.model.grid['lam'] # convenience


        # --- define spectra produced by the code

        self.stellar = core.sed(self.lam, description = 'intrinsic (stellar) SED') # includes no reprocessing by gas or dust
        self.intrinsic = core.sed(self.lam, description = 'SED with nebular emission (HII) but no BC or ISM dust')
        self.BC = core.sed(self.lam, description = 'SED with nebular emission (HII), BC dust, but no ISM dust')
        self.no_HII_no_BC = core.sed(self.lam, description = 'SED with no nebular emission (HII) or dusty BC (same as setting fesc=1.0)')
        self.total = core.sed(self.lam, description = 'SED including both birth cloud and ISM dust')

        if isinstance(tauVs_ISM, np.ndarray) and not self.model.dust_ISM:
            print("WARNING! You have supplied ISM optical depths but have not specificied a ISM dust model. This doesn't sound right!")
        if isinstance(tauVs_BC, np.ndarray) and not self.model.dust_BC:
            print("WARNING! You have supplied BC optical depths but have not specificied a BC dust model. This doesn't sound right!")

        if not isinstance(tauVs_ISM, np.ndarray): tauVs_ISM = np.zeros(Masses.shape)
        if not isinstance(tauVs_BC, np.ndarray): tauVs_BC = np.zeros(Masses.shape)

        if self.model.dust_ISM:

            dust_model, dust_model_params = self.model.dust_ISM
            tau_ISM_lam = getattr(dust_curves, dust_model)(params = dust_model_params).tau(self.lam) # optical depth relative to tau_V for wavelengths

        if self.model.dust_BC:

            dust_model, dust_model_params = self.model.dust_BC
            tau_BC_lam = getattr(dust_curves, dust_model)(params = dust_model_params).tau(self.lam) # optical depth relative to tau_V for wavelengths


        for Mass, Age, Metallicity, tauV_ISM, tauV_BC in zip(Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC):

            log10age = np.log10(Age) + 6. # log10(age/yr)
            log10Z = np.log10(Metallicity) # log10(Z)

            # --- determine closest SED grid point

            ia = (np.abs(self.model.grid['log10age'] - log10age)).argmin()
            iZ = (np.abs(self.model.grid['log10Z'] - log10Z)).argmin()

            sed = empty() # SED for this star particle only

            sed.stellar = Mass * self.model.grid['stellar_incident'][ia, iZ]

            if log10age < log10t_BC:

                sed.intrinsic = Mass * (fesc*self.model.grid['stellar_incident'][ia, iZ] + (1.-fesc)*(self.model.grid['stellar_transmitted'][ia, iZ] + self.model.grid['nebular'][ia, iZ]))

                # --- determine BC dust attenuation

                if self.model.dust_BC:
                    tau = tauV_BC * tau_BC_lam
                    T = np.exp(-tau)
                else:
                    T = 1.0

                sed.BC = Mass * (fesc*self.model.grid['stellar_incident'][ia, iZ] + T*(1.-fesc)*(self.model.grid['stellar_transmitted'][ia, iZ] + self.model.grid['nebular'][ia, iZ]))

            else:

                sed.intrinsic = Mass * self.model.grid['stellar_incident'][ia, iZ]
                sed.BC = Mass * self.model.grid['stellar_incident'][ia, iZ] # no additional contribution from birth cloud

            # --- determine ISM dust attenuation

            if self.model.dust_ISM:
                tau = tauV_ISM * tau_ISM_lam
                T = np.exp(-tau)
            else:
                T = 1.0

            self.stellar.lnu += sed.stellar
            self.intrinsic.lnu += sed.intrinsic
            self.BC.lnu += sed.BC
            self.total.lnu += T * sed.BC
            self.no_HII_no_BC.lnu += T * sed.stellar

            self.f_esc = self.total.lnu/self.intrinsic.lnu
            self.tau = -np.log(self.f_esc)
            self.tau_relV = self.tau/np.interp(5500.,self.lam,self.tau)











class EmissionLines():

    def __init__(self, SPSIMF, dust_ISM = False, dust_BC = False, verbose = True, path_to_SPS_grid = f'{FLARE_dir}/data/SPS/nebular/3.0/'):

        self.SPSIMF = SPSIMF

        self.grid = pickle.load(open(f'{path_to_SPS_grid}/{SPSIMF}/lines.p','rb'), encoding='latin1')  # --- open grid

        print(self.grid['HI3750'].keys())

        self.lines = self.grid['lines']


        self.lam = {l: self.grid[l]['lam'] for l in self.lines}

        self.lams = np.array([self.lam[l] for l in self.lines])

        if verbose:
            print('Available lines:')
            for lam,line in sorted(zip(self.lams,self.lines)): print(f'{line}')

        self.dust_BC = dust_BC
        self.dust_ISM = dust_ISM

        if self.dust_ISM:
            dust_model, dust_model_params = self.dust_ISM
            self.dust_curve_ISM = getattr(dust_curves, dust_model)(params = dust_model_params)

        if self.dust_BC:
            dust_model, dust_model_params = self.dust_BC
            self.dust_curve_BC = getattr(dust_curves, dust_model)(params = dust_model_params)

        self.units = {'luminosity': 'erg/s', 'nebular_continuum': 'erg/s/Hz', 'stellar_incident_continuum': 'erg/s/Hz', 'stellar_transmitted_continuum': 'erg/s/Hz', 'continuum': 'erg/s/Hz', 'EW': 'AA'}

        if 'stellar_transmitted_continuum' not in self.grid[self.lines[0]]:
            # --- this is now the canonical way of doing things
            for l in self.lines:
                self.grid[l]['stellar_transmitted_continuum'] = self.grid[l]['stellar_continuum']
                self.grid[l]['stellar_incident_continuum'] = self.grid[l]['stellar_continuum']



    def get_line_luminosities(self, lines, Masses, Ages, Metallicities, tauVs_ISM = False, tauVs_BC = False, fesc = 0.0, log10t_BC = 7., verbose = False):

        print('--- calculate quantities for multiple emission lines')
        print(lines)
        o = {}
        for line in lines:
            o[line] = self.get_line_luminosity(line, Masses, Ages, Metallicities, tauVs_ISM = tauVs_ISM, tauVs_BC = tauVs_BC, fesc = fesc, log10t_BC = log10t_BC, verbose = verbose)

        return o


    def get_line_luminosity(self, line, Masses, Ages, Metallicities, tauVs_ISM = False, tauVs_BC = False, fesc = 0.0, log10t_BC = 7., verbose = False):

        line = line.split(',')

        if type(tauVs_ISM) is not np.ndarray:
            tauVs_ISM = np.zeros(Masses.shape)
            if verbose: 'WARNING: no ISM optical depths provided, quantities will not include ISM attenuation!'

        if type(tauVs_BC) is not np.ndarray:
            tauVs_BC = np.zeros(Masses.shape)
            if verbose: 'WARNING: no BC optical depths provided, quantities will not include BC attenuation!'

        if not self.dust_ISM:
            if verbose: 'WARNING: no ISM dust model specified, quantities will not include ISM attenuation!'

        if not self.dust_BC:
            if verbose: 'WARNING: no BC dust model specified, quantities will not include ISM attenuation!'


        lam = np.mean([self.grid[l]['lam'] for l in line])

        if verbose:
            print(f'----- {line}')
            print(f'line wavelength/\AA: {lam}')

        o = {t: 0.0 for t in ['luminosity','continuum']} # output dictionary

        for Mass, Age, Metallicity, tauV_BC, tauV_ISM in zip(Masses, Ages, Metallicities, tauVs_BC, tauVs_ISM):

            log10age = np.log10(Age) + 6. # log10(age/yr)
            log10Z = np.log10(Metallicity) # log10(Z)

            # --- determine closest SED grid point
            ia = (np.abs(self.grid['log10age'] - log10age)).argmin()
            iZ = (np.abs(self.grid['log10Z'] - log10Z)).argmin()

            # --- determine ISM dust attenuation
            if self.dust_ISM:
                tau = tauV_ISM * self.dust_curve_ISM.tau(lam)
                T_ISM = np.exp(-tau)
            else:
                T_ISM = 1.0


            if log10age < log10t_BC:

                if self.dust_BC:
                    tau = tauV_BC * self.dust_curve_BC.tau(lam)
                    T_BC = np.exp(-tau)
                else:
                    T_BC = 1.0

                for l in line:
                    o['luminosity'] += Mass * T_BC * T_ISM * (1.-fesc) * 10**self.grid[l]['luminosity'][ia, iZ] # erg/s
                    o['continuum'] += Mass * (T_ISM*fesc*self.grid[l]['stellar_incident_continuum'][ia, iZ] + (T_ISM*T_BC)*(1-fesc)*(self.grid[l]['stellar_transmitted_continuum'][ia, iZ] + self.grid[l]['nebular_continuum'][ia, iZ])) # erg/s

            else:
                for l in line:
                    o['continuum'] += Mass * T_ISM * self.grid[l]['stellar_incident_continuum'][ia, iZ]



        total_continuum = (o['continuum']/float(len(line)))*(3E8)/((lam/float(len(line)))**2*1E-10)

        o['EW'] = o['luminosity']/total_continuum

        if verbose:
            for k,v in o.items(): print(f'log10({k}/{self.units[k]}): {np.log10(v):.2f}')

        return o
