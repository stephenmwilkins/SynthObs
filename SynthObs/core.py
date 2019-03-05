
import numpy as np

import os

import FLARE

FLARE_dir = FLARE.FLARE_dir

class empty: pass 


def test_data(path_to_example_data = FLARE_dir + 'data/package_example_data/SynthObs/'):

    data = empty()
    
    data.X = np.load(path_to_example_data+'/X.npy')
    data.Y = np.load(path_to_example_data+'/Y.npy')
    data.MetSurfaceDensities = np.load(path_to_example_data+'/MetSurfaceDensities.npy') # surface density of metals along line of sight to each star particle. Used for calculating dust.
    data.Ages = np.load(path_to_example_data+'/Ages.npy') # age of each star particle in Myr 
    data.Metallicities = np.load(path_to_example_data+'/Metallicities.npy') # mass fraction of stars in metals (Z)
    data.Masses = np.ones(data.Ages.shape) * 1E10*5.90556119E-05/0.697 # mass of each star particle in M_sol . NOTE: this value is for BLUETIDES in h-less units.

    return data


class get_object_data():

    def __init__(self, snap, ID, subID, data_path = FLARE_dir + 'simulations/BlueTides/all'):
    
        path = data_path + '/{snap}/{ID}/{subID}/'.format(snap=snap,ID=ID,subID=subID)
    
        self.X = np.load(path+'X.npy')
        self.Y = np.load(path+'Y.npy')
        self.MetSurfaceDensities = np.load(path+'MetSurfaceDensities.npy') # surface density of metals along line of sight to each star particle. Used for calculating dust.
        self.Ages = np.load(path+'Ages.npy') # age of each star particle in Myr 
        self.Metallicities = np.load(path+'Metallicities.npy') # mass fraction of stars in metals (Z)
        self.Masses = np.ones(self.Ages.shape) * 1E10*5.90556119E-05/0.697 # mass of each star particle in M_sol . NOTE: this value is for BLUETIDES in h-less units.





class all_test_data():

    def __init__(self, snap, path_to_example_data = FLARE_dir + 'data/package_example_data/SynthObs/'):
    
        self.data_path = path_to_example_data
    
        self.obj = []
        
        for x in os.walk(path_to_example_data + snap):
            if x[0].split('/')[-3]==snap: self.obj.append('/'.join(x[0].split('/')[-3:]))
        
        self.M = np.array([len(np.load(self.data_path+o+'/Ages.npy'))*1E10*5.90556119E-05/0.697 for o in self.obj])      
        
        sorted = np.argsort(self.M)
        self.M = self.M[sorted]
        self.obj = np.array(self.obj)[sorted]
        
        self.Mr = np.round(np.log10(self.M),1)
        
        self.N = len(self.obj)
        
        self.j = 0
          
    def get(self, i):
    
        path = self.data_path + self.obj[i]+'/'
    
        data = empty()
    
        data.id = self.obj[i]
        data.X = np.load(path+'X.npy')
        data.Y = np.load(path+'Y.npy')
        data.MetSurfaceDensities = np.load(path+'MetSurfaceDensities.npy') # surface density of metals along line of sight to each star particle. Used for calculating dust.
        data.Ages = np.load(path+'Ages.npy') # age of each star particle in Myr 
        data.Metallicities = np.load(path+'Metallicities.npy') # mass fraction of stars in metals (Z)
        data.Masses = np.ones(data.Ages.shape) * 1E10*5.90556119E-05/0.697 # mass of each star particle in M_sol . NOTE: this value is for BLUETIDES in h-less units.

        return data
        
    def next(self):
        self.j += 1
        return self.get(self.j - 1) 
    