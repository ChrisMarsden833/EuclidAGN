# Libraries
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import stats
from scipy import special
import pandas as pd
import os
import re
import multiprocessing
from numba import jit
from math import pi

# Specific Libraries
from colossus.cosmology import cosmology
from colossus.lss import mass_function

def PlotHaloMassFunction(haloes, z, volume, cosmo, path):
    """Function to plot HMF and compares it to the Colossus HMF. 

    Attributes
        haloes (np array, floats) : array of halo masses, in log10
        redshift 
    """
    h = cosmo.H0/100

    haloes = np.log10(haloes)

    # First plot the actual HMF
    width = 0.1
    bins = np.arange(10, 15, width)
    hist = np.histogram(haloes, bins = bins)[0]
    hmf = (hist/(volume))/(width)
    fig = plt.figure()
    plt.plot(10**bins[0:-1], hmf, 'o', label = "Haloes Read")

    # Get the HMF from colossus.
    try: 
        binwidth = 0.01
        M = 10**np.arange(10.0, 15.0, binwidth) + np.log10(h) # In unit's of h, for now.
        mfunc = mass_function.massFunction(M*h, z, mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')*np.log(10) *(h**3) #dn/dlog10M
        plt.plot(M, mfunc, label = "Colossus")
    except:
        print("Colossus Failed to plot HMF")
    
    #np.savetxt("HMF.txt", np.c_[np.log10(10**bins[0:-1]), np.log10(hmf)] )

    plt.xlabel("Halo Mass $M_\odot/h$")
    plt.ylabel(r'$d\phi /d(log\;L_x)\;[Mpc^{-3}]/h$')
    plt.title("Halo Mass function from Multidark (centrals), z = {}".format(z))
    plt.loglog()
    plt.legend()
    # Write the file to the approprate location.
    savePath = path + 'HMF_Comparison.png'
    fig.savefig(savePath)
    plt.close()