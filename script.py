import AGNCatalogToolbox as act
import ACTTestingEncapsulation as actt

import numpy as np
from matplotlib import pyplot as plt

# Make sure we have access to the higher level directory.
import sys
sys.path.insert(0, '../')

# Set the redshift of interest here.
redshift = 0.0

obj = actt.AGNCatalog()
obj.set_z(redshift)
# Load in the MultiDark Halos.
obj.load_dm_catalog()
obj.assign_stellar_mass()
obj.assign_black_hole_mass()
obj.assign_duty_cycle()
obj.assign_luminosity()
