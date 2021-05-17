

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import synthobs
import synthobs.morph
from synthobs.morph import measure



test = synthobs.test_data() # --- read in some test data


# ------ Make rest-frame luminosity image


for resolution, smoothing in zip([0.05, 0.1, 0.1], ['adaptive', 'adaptive', 'simple']):


    img = synthobs.morph.physical_image(test.X, test.Y, test.Masses, smoothing = smoothing, resolution = resolution)

    m = measure.intrinsic(img)
    m.detect_sources()

    r_e = m.r_e()
    r_e['simple'] = measure.simple(test.X, test.Y, test.Masses)[0]

    print(resolution, smoothing, r_e) # --- measure effective_radius in several different ways
