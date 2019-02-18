
import numpy as np


class test_data():

    def __init__(self, path_to_example_data = '/Users/stephenwilkins/Dropbox/Research/Data/package_example_data/SynthObs/'):
    
        self.X = np.load(path_to_example_data+'/X.npy')
        self.Y = np.load(path_to_example_data+'/Y.npy')
        self.MetSurfaceDensities = np.load(path_to_example_data+'/MetSurfaceDensities.npy') # surface density of metals along line of sight to each star particle. Used for calculating dust.
        self.Ages = np.load(path_to_example_data+'/Ages.npy') # age of each star particle in Myr 
        self.Metallicities = np.load(path_to_example_data+'/Metallicities.npy') # mass fraction of stars in metals (Z)
        self.Masses = np.ones(self.Ages.shape) * 1E10*5.90556119E-05/0.697 # mass of each star particle in M_sol . NOTE: this value is for BLUETIDES in h-less units.
