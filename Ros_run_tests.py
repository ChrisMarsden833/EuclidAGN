import os
import glob
from Ros_SFR_test_pars import simulation
import multiprocessing as mp
#from multiprocessing import Process, Queue


curr_dir=os.getcwd()

################################
# import simulation parameters
sub_dir= 'Sahu_Schechter/' 
sub_dir= 'Davis_Schechter/'
sub_dir='Sahu_Gaussian/' # z=1: pars1. z=2.7 :pars2. z=0.45:pars3 
sub_dir= 'Test_Marconi/' # up to pars7
sub_dir= 'Standard_PowerLaw/'
sub_dir= 'Standard_Gaussian/'
sub_dir= 'Standard_Schechter/' # up to pars7 
sub_dir= 'R&V_Gaussian_slope2/' 
sub_dir='R&V_Gaussian/' # z=1: pars1. z=2.7 :pars2. z=0.45:pars3. z=1 varying sigma: pars4
sub_dir= 'R&V_Schechter/'
#sys.path.append(curr_dir+'/Ros_plots/'+sub_dir)

var_files = sorted(glob.glob(curr_dir+'/Ros_plots/'+sub_dir+'pars*.py'))

def main():
  pool = mp.Pool(mp.cpu_count())
  pool.map(simulation, var_files)

if __name__ == "__main__":
  main()