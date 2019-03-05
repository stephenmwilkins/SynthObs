

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import FLARE.filters
import SynthObs
import SynthObs.Morph 
from SynthObs.SED import models
from SynthObs.Morph import measure 





Ndim = 75


model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
model.dust = {'A': 5.2, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model

intrinsic_model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
intrinsic_model.dust = False # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model



filters = FLARE.filters.FAKE

# filters = ['FAKE.FAKE.'+f for f in ['1500','Hth']]

F = FLARE.filters.add_filters(filters, new_lam = model.lam) 


model.create_Lnu_grid(F)
intrinsic_model.create_Lnu_grid(F)

test = SynthObs.test_data() # --- read in some test data

L = {f: models.generate_Lnu_array(model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters}

L.update({'intrinsic_'+f: models.generate_Lnu_array(intrinsic_model, test.Masses, test.Ages, test.Metallicities, test.MetSurfaceDensities, F, f) for f in filters})



# ------ Make rest-frame luminosity image

IMGs = SynthObs.Morph.physical_images(test.X, test.Y, L, L.keys(), Ndim = Ndim)


fig, axes = plt.subplots(2, len(filters), figsize = (len(filters)*2., 4))
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

mx = np.max([np.max(IMGs[f].img) for f in filters])
mx_intrinsic = np.max([np.max(IMGs['intrinsic_'+f].img) for f in filters])

for i, f in enumerate(filters):
   
    axes[0,i].imshow(IMGs[f].img, vmin = 0.0, vmax = mx)  
    axes[0,i].get_xaxis().set_ticks([])
    axes[0,i].get_yaxis().set_ticks([])

    # -- scaling tests
    
#     max = np.max(IMGs[f].img)
#     img = IMGs[f].img/max
#     lim = 1E-2
#     img[img<lim] = lim
#     img = np.log10(img)
#     vmax = 0.0
#     vmin = np.log10(lim)
#     axes[1,i].imshow(img, vmin = vmin, vmax = vmax)  
#     
#     max = np.max(IMGs[f].img)
#     img = IMGs[f].img/max
#     img *= 1
#     img = np.arcsinh(img)
#     axes[1,i].imshow(img)  
    
    
    axes[1,i].imshow(IMGs['intrinsic_'+f].img, vmin = 0.0, vmax = mx_intrinsic)  
    axes[1,i].get_xaxis().set_ticks([])
    axes[1,i].get_yaxis().set_ticks([])
   
    print(f, np.sum(IMGs['intrinsic_'+f].img), np.sum(IMGs[f].img))
    
    
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


    axes[0,i].axhline(Ndim/2.+0.5, c='1.0', lw=1, alpha = 0.2)
    axes[0,i].axvline(Ndim/2.+0.5, c='1.0', lw=1, alpha = 0.2)
    axes[1,i].axhline(Ndim/2.+0.5, c='1.0', lw=1, alpha = 0.2)
    axes[1,i].axvline(Ndim/2.+0.5, c='1.0', lw=1, alpha = 0.2)

    info = r'$\rm log_{{10}}(L_{{\nu}}/erg\ s^{{-1}}\ Hz^{{-1}})={0:9.1f}$'.format(np.log10(np.sum(IMGs[f].img)))

    axes[0,i].text(0.5, 0.1, info, fontsize = 7, color='1.0', alpha = 1.0, horizontalalignment='center', verticalalignment='center', transform=axes[0,i].transAxes)

    info = r'$\rm log_{{10}}(L_{{\nu}}/erg\ s^{{-1}}\ Hz^{{-1}})={0:9.1f}$'.format(np.log10(np.sum(IMGs['intrinsic_'+f].img)))

    axes[1,i].text(0.5, 0.1, info, fontsize = 7, color='1.0', alpha = 1.0, horizontalalignment='center', verticalalignment='center', transform=axes[1,i].transAxes)



plt.show()




# ------ make RGB image



RGB = np.zeros((IMGs['FAKE.FAKE.1500'].img.shape[0],IMGs['FAKE.FAKE.1500'].img.shape[1], 3), dtype=float)
RGB[:,:,2] = IMGs['FAKE.FAKE.1500'].img/mx
RGB[:,:,1] = IMGs['FAKE.FAKE.Vth'].img/mx
RGB[:,:,0] = IMGs['FAKE.FAKE.Hth'].img/mx



fig = plt.figure(figsize = (4,4))
ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), **{'xticks': [], 'yticks': []})       


ax.imshow(RGB, interpolation='nearest', origin='low')  

plt.show()
fig.clf()






