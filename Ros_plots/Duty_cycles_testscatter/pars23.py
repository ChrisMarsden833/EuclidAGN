# set pars for z=0.45, Reines&Volonteri15, testing duty cycle: constant =0.2

import numpy as np

################################
# set simulation parameters
z = 0.45
reds_dic={0.45:0, 1:1, 1.7:2, 2.7:3}
index=reds_dic.get(z) # needed for IDL data

methods={'halo_to_stars':'Grylls19', # 'Grylls19' or 'Moster'
    'BH_mass_method':"Reines&Volonteri15", #"Shankar16", "KormendyHo", "Eq4", "Davis18", "Sahu19" and "Reines&Volonteri15"
    'BH_mass_scatter':1., # "Intrinsic" or float
    'duty_cycle':0.1, # "Schulze", "Man16", "Geo" or float (0.2)
    'edd_ratio':"Gaussian", # "Schechter", "PowerLaw", "Gaussian", "Geo"
    'bol_corr':'Lusso12_modif', # 'Duras20', 'Marconi04', 'Lusso12_modif'
    'SFR':'Carraro20' # 'Tomczak16', "Schreiber15", "Carraro20"
    }

# looping on:
if methods['edd_ratio']=="Schechter":
   variable_name = r"$\lambda$"
   par_str= 'lambda'
   #variable_name = r"$\alpha$"
   #par_str= 'alpha'
elif methods['edd_ratio']=="Gaussian":
   #variable_name = r"$\sigma$"
   #par_str= 'sigma'
   variable_name = r"$\mu$"
   par_str= 'mean'
parameters = [-1.0,-0.5,0.,0.5,1.0]

################################
# Edd ratio parameters definition:
sigma_z=0.3
mu_z=-2.0

if methods['edd_ratio']=='Gaussian':
   lambda_z=sigma_z
   alpha_z=mu_z

if methods['edd_ratio']=="Schechter":
   print(f'lambda_z={lambda_z}, alpha_z={alpha_z}')
elif methods['edd_ratio']=="Gaussian":
   print(f'sigma={sigma_z}, mu_z={mu_z}')


################################
# mass range restrictions
M_inf=0
M_sup=0
if z==2.7:
    M_inf=10.
elif methods['BH_mass_method']=="Shankar16":
    M_inf=10
elif methods['BH_mass_method']=="Davis18":
    M_inf=10.3
    M_sup=11.4
elif methods['BH_mass_method']=="Sahu19":
    M_inf=10.
    M_sup=12.15
elif methods['BH_mass_method']=="Reines&Volonteri15":
    M_inf=9.
print(M_inf,M_sup)

################################
# filename suffix
suffix=f"_const{methods['duty_cycle']}_scatter{methods['BH_mass_scatter']}"
