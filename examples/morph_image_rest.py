

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import FLARE.filters
import SynthObs
import SynthObs.Morph.images as images
from SynthObs.SED import models
from SynthObs.Morph import measure 




resolution = 0.2
ndim = 50
smoothing = ('adaptive', 16)
smoothing = ('convolved_gaussian', 0.2)

model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model

intrinsic_model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
intrinsic_model.dust = False # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model



filters = FLARE.filters.FAKE

F = FLARE.filters.add_filters(filters, new_lam = model.lam) 


model.create_Lnu_grid(F)
intrinsic_model.create_Lnu_grid(F)

test = SynthObs.test_data() # --- read in some test data

L = {f: models.generate_Lnu_array(model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters}

L.update({'intrinsic_'+f: models.generate_Lnu_array(intrinsic_model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters})



# ------ Make rest-frame luminosity image

imgs = {f: images.core(test.X, test.Y, L[f], resolution = resolution, ndim = ndim, smoothing = smoothing, verbose = False) for f in filters}
imgs.update({'intrinsic_'+f: images.core(test.X, test.Y, L['intrinsic_'+f], resolution = resolution, ndim = ndim, smoothing = smoothing, verbose = False) for f in filters})



fig, axes = plt.subplots(2, len(filters), figsize = (len(filters)*2., 4))
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

mx = np.max([np.max(imgs[f].img) for f in filters])
mx_intrinsic = np.max([np.max(imgs['intrinsic_'+f].img) for f in filters])

for i, f in enumerate(filters):
   
    axes[0,i].imshow(imgs[f].img, vmin = 0.0, vmax = mx)  
    axes[0,i].get_xaxis().set_ticks([])
    axes[0,i].get_yaxis().set_ticks([])
    
    axes[1,i].imshow(imgs['intrinsic_'+f].img, vmin = 0.0, vmax = mx_intrinsic)  
    axes[1,i].get_xaxis().set_ticks([])
    axes[1,i].get_yaxis().set_ticks([])
   
    print('{0}: L_int = {1:.2f} L_obs = {2:.2f}'.format(f, np.log10(np.sum(imgs['intrinsic_'+f].img)), np.log10(np.sum(imgs[f].img))))
    
    
    # --- add labels and guide lines
    
    fn = f.split('.')[-1]
    
    if fn == '1500':
        label = 'FUV'
    elif fn == '2500':
        label = 'NUV'
    else:
        label = fn[0]

        
    axes[0,i].text(0.5, 0.9, r'$\rm\bf{0}$'.format(label), fontsize = 10, color='1.0', alpha = 1.0, horizontalalignment='center', verticalalignment='center', transform=axes[0,i].transAxes)

    axes[1,i].text(0.5, 0.9, r'$\rm Intrinsic\ \bf{0}$'.format(label), fontsize = 10, color='1.0', alpha = 1.0, horizontalalignment='center', verticalalignment='center', transform=axes[1,i].transAxes)


    axes[0,i].axhline(ndim/2.+0.5, c='1.0', lw=1, alpha = 0.2)
    axes[0,i].axvline(ndim/2.+0.5, c='1.0', lw=1, alpha = 0.2)
    axes[1,i].axhline(ndim/2.+0.5, c='1.0', lw=1, alpha = 0.2)
    axes[1,i].axvline(ndim/2.+0.5, c='1.0', lw=1, alpha = 0.2)

    info = r'$\rm log_{{10}}(L_{{\nu}}/erg\ s^{{-1}}\ Hz^{{-1}})={0:9.1f}$'.format(np.log10(np.sum(imgs[f].img)))

    axes[0,i].text(0.5, 0.1, info, fontsize = 7, color='1.0', alpha = 1.0, horizontalalignment='center', verticalalignment='center', transform=axes[0,i].transAxes)

    info = r'$\rm log_{{10}}(L_{{\nu}}/erg\ s^{{-1}}\ Hz^{{-1}})={0:9.1f}$'.format(np.log10(np.sum(imgs['intrinsic_'+f].img)))

    axes[1,i].text(0.5, 0.1, info, fontsize = 7, color='1.0', alpha = 1.0, horizontalalignment='center', verticalalignment='center', transform=axes[1,i].transAxes)



plt.show()
fig.clf()



# ------ make RGB image



RGB = np.zeros((imgs['FAKE.FAKE.1500'].img.shape[0],imgs['FAKE.FAKE.1500'].img.shape[1], 3), dtype=float)
RGB[:,:,2] = imgs['FAKE.FAKE.1500'].img/mx
RGB[:,:,1] = imgs['FAKE.FAKE.Vth'].img/mx
RGB[:,:,0] = imgs['FAKE.FAKE.Hth'].img/mx



fig = plt.figure(figsize = (4,4))
ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), **{'xticks': [], 'yticks': []})       


ax.imshow(RGB, interpolation='nearest', origin='low')  

plt.show()
fig.clf()






