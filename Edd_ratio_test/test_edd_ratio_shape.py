#%%
"""
Test Eddington ratio distribution shapes
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
# Edd ratio values range
edd_bin = np.arange(-4, 0, 0.001)
edd=10 ** edd_bin

def schechter(edd,arg1,arg2):
   prob = ((edd / (10. ** arg1)) ** arg2)
   return prob* np.exp(-(edd / (10. ** arg1)))

def gaussian(edd,arg1,arg2):
   return np.exp(-(edd - arg2) ** 2. / (2.*arg1 ** 2.))
#%%
# Shankar + Schechter z=1
arg1=-0.8
arg2= 1.6

plt.plot(edd,schechter(edd,arg1,arg2));
#%%
# shankar + gaussian
arg1= 0.2
arg2= 0.25

plt.plot(edd,gaussian(edd,arg1,arg2));
#%%
# shankar + gaussian
arg1= 0.2
arg2= 0.6

plt.plot(edd,gaussian(edd,arg1,arg2));
# %%
