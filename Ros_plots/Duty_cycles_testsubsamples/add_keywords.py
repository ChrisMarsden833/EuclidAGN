import glob
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/users/Ros/Google_Drive/Valparaiso/LX_SFR_causality/EuclidAGN2/AGNCatalogToolbox/')
from Ros_utilities import append_new_line

#import os
#curr_dir=os.getcwd()

par_files=glob.glob(f'./Ros_plots/Duty_cycles_testsubsamples/pars*.py') 
print(par_files)

# sf_subsamples=False
# AGN_extraction=False
# weighted_luminosity=True

for file in par_files:
   append_new_line(file,'')
   append_new_line(file,'weighted_luminosity=False')
   append_new_line(file,'sf_subsamples=True')
   append_new_line(file,'AGN_extraction=True')