

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import SynthObs
import SynthObs.Morph.images  


redshift = 8.
h = 0.697
resolution = 0.1
Ndim = 50
smoothing = 'gaussian'
# smoothing = False
smoothing_length = (1.5/h)/(1.+redshift)
# smoothing_length = 0.1




test = SynthObs.test_data() # --- read in some test data


# ------ high-resolution image

# highres = SynthObs.Morph.images.physical_individual(test.X, test.Y, test.Masses, resolution = 0.05, Ndim = Ndim, smoothing = smoothing, smoothing_length = smoothing_length)
# plt.imshow(highres.data)
# plt.show()


# ------ compare adaptive and simple smoothing

# adaptive = SynthObs.Morph.images.physical_individual(test.X, test.Y, test.Masses, Ndim = 50, smoothing = 'adaptive')
# simple = SynthObs.Morph.images.physical_individual(test.X, test.Y, test.Masses, Ndim = 50)
# 
# print(np.sum(adaptive.data), np.sum(simple.data))
# 
# R = adaptive.data-simple.data
# 
# print(np.max(R)/np.max(simple.data))
# 
# plt.imshow(adaptive.data)
# plt.show()
# plt.imshow(simple.data)
# plt.show()
# plt.imshow(R)
# plt.show()




# ------ Make stellar mass vs. recent SF comparison


imgs = {}

imgs['mass'] = SynthObs.Morph.images.physical_individual(test.X, test.Y, test.Masses, resolution = resolution, Ndim = Ndim, smoothing = smoothing, smoothing_length = smoothing_length)

s = test.Ages<10.
imgs['sfr'] = SynthObs.Morph.images.physical_individual(test.X[s], test.Y[s], test.Masses[s], resolution = resolution, Ndim = Ndim, smoothing = smoothing, smoothing_length = smoothing_length)

N = len(imgs.keys())
fig, axes = plt.subplots(1, N, figsize = (N*2., 2))
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

mass_label = r'$\rm log_{{10}}(M^{{*}}/M_{{\odot}})={0:9.2f}$'.format(np.log10(np.sum(imgs['mass'].data)))
sf_label = r'$\rm SFR_{{10}}/(M_{{\odot}}\ yr^{{-1}})={0:9.1f}$'.format(np.sum(imgs['sfr'].data)/1E7)

for i, f, label, cm, info in zip(range(2), ['mass', 'sfr'], [r'$\rm\bf M^{\star}$', r'$\rm\bf recent\ SF$'], ['viridis','inferno'], [mass_label, sf_label]):   
    axes[i].imshow(imgs[f].data, cmap = cm)   
    axes[i].get_xaxis().set_ticks([])
    axes[i].get_yaxis().set_ticks([])
    axes[i].text(0.5, 0.9, f, fontsize = 10, color='1.0', alpha = 1.0, horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
    axes[i].axhline(Ndim-0.5, c='1.0', lw=1, alpha = 0.2)
    axes[i].axvline(Ndim-0.5, c='1.0', lw=1, alpha = 0.2)
    axes[i].text(0.5, 0.1, info, fontsize = 7, color='1.0', alpha = 1.0, horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)



plt.show()



