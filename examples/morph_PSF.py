

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flare
import flare.observatories
import synthobs
import synthobs.morph.PSF

from photutils import CircularAperture
from photutils import aperture_photometry

logged = True

base_width = 10. # "
sampling = 10

filters = ['Euclid.NISP.H', 'Hubble.WFC3.f160w', 'Webb.NIRCam.F150W'] #

# filters = ['Spitzer.IRAC.ch1', 'JWST.NIRCAM.F356W'] #

PSFs = synthobs.morph.PSF.PSFs(filters)

print('width: {0}/"'.format(base_width))
print('sampling: {0}/"'.format(sampling))

fig, axes = plt.subplots(2, len(filters), figsize = (len(filters)*2., 4))
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)



make_profile = True

if make_profile:
    fig_profile = plt.figure()
    ax_profile = fig_profile.add_axes((0.15, 0.15, 0.8, 0.8))



for i, f in enumerate(filters):

    print('-'*10, f)
    print('native pixel scale: {0}/"'.format(flare.observatories.filter_info[f]['pixel_scale']))

    width_pix_sampled = base_width // (flare.observatories.filter_info[f]['pixel_scale'] / sampling) # width of sampled pixels -> want to be odd

    if width_pix_sampled % 2 == 0: width_pix_sampled -= 1

    print('width: {0}/sampled pix'.format(width_pix_sampled))

    width_pix = width_pix_sampled // sampling


    xx = yy = np.linspace(-width_pix/2., width_pix/2. , int(width_pix))
    img = PSFs[f].f((xx-0.0), (yy-0.0))
    img /= np.sum(img)

    xx = yy = np.linspace(-width_pix/2., width_pix/2. , int(width_pix_sampled))  # sub sampled
    img_sub = PSFs[f].f((xx-0.0), (yy-0.0))
    img_sub /= np.sum(img_sub)


    # --- plot 2D PSF

    if logged:
        axes[1,i].imshow(np.log10(img/np.max(img) + 1E-4), interpolation = 'nearest', vmin = -4, vmax = 0.0)
        axes[0,i].imshow(np.log10(img_sub/np.max(img_sub) + 1E-4), interpolation = 'nearest', vmin = -4, vmax = 0.0)
    else:
        axes[1,i].imshow(img, interpolation = 'nearest')
        axes[0,i].imshow(img_sub, interpolation = 'nearest')

    for j in range(2):
        axes[j,i,].get_xaxis().set_ticks([])
        axes[j,i].get_yaxis().set_ticks([])

    axes[0,i].text(0.5, 0.85, f.split('.')[-1], fontsize = 15, color='1.0', alpha = 0.3, horizontalalignment='center', verticalalignment='center', transform=axes[0,i].transAxes)


    # --- measure PSF profile


    if make_profile:

        Ndim = img_sub.shape[0]

        centre = (Ndim//2, Ndim//2)

        radii_pix = np.arange(1, 500., 1)
        radii_arcsec = radii_pix * (flare.observatories.filter_info[f]['pixel_scale'] / sampling)

        apertures = [CircularAperture(centre, r=r) for r in radii_pix] #r in pixels

        phot_table = aperture_photometry(img_sub, apertures)

        frac = np.array([phot_table['aperture_sum_{0}'.format(i)][0] for i, r in enumerate(radii_pix)])

        for efrac in [0.5, 0.8]:
            print('EE(f={0}): {1:0.2f}"'.format(efrac, np.interp(efrac, frac, radii_arcsec)))

        for r in [0.35/2.]:
            print('EE(r<{0}): {1:0.2f}'.format(r, np.interp(r, radii_arcsec, frac)))

        ax_profile.plot(radii_arcsec, frac, label = f)




fig.savefig('figs/morph_PSF_2D.pdf')
fig.clf()


if make_profile:
    fig_profile.legend()
    fig_profile.savefig('figs/morph_PSF_COG.pdf')
    fig_profile.clf()
