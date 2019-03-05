

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

IMGs = SynthObs.Morph.physical_images(test.X, test.Y, L, L.keys(), Ndim = 100)




fig, axes = plt.subplots(2, len(filters), figsize = (len(filters)*2., 4))
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)


mx = np.max([np.max(IMGs[f].img) for f in filters])
mx_intrinsic = np.max([np.max(IMGs['intrinsic_'+f].img) for f in filters])

print(mx,mx_intrinsic)


for i, f in enumerate(filters):
   
   
   
    axes[0,i].imshow(IMGs[f].img, vmin = 0.0, vmax = mx)  
    axes[0,i].get_xaxis().set_ticks([])
    axes[0,i].get_yaxis().set_ticks([])
    
    axes[1,i].imshow(IMGs['intrinsic_'+f].img, vmin = 0.0, vmax = mx_intrinsic)  
    axes[1,i].get_xaxis().set_ticks([])
    axes[1,i].get_yaxis().set_ticks([])
   
   
    print(f, np.sum(IMGs['intrinsic_'+f].img), np.sum(IMGs[f].img))
    
    axes[0,i].text(0.5, 0.9, f.split('.')[-1], fontsize = 30, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=axes[0,i].transAxes)


fig.savefig('rest.png')
plt.show()



