

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import synthobs



# -- demonstration of how to access additional test objects

test_data = synthobs.all_test_data('086') # --- read in some test data

for i in range(test_data.N):

    # data = test_data.get(i)
    data = test_data.next()

    print(i, data.id, np.log10(len(data.X)* 1E10*5.90556119E-05/0.697))
