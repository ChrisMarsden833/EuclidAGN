# set pars for z=1, Reines&Volonteri15 with norm==9.5, log gaussian mean=-1.75 with sigma=0.3dex

import numpy as np

################################
# set simulation parameters
z = 1.0
reds_dic={0.45:0, 1:1, 1.7:2, 2.7:3}
index=reds_dic.get(z) # needed for IDL data

methods={'halo_to_stars':'Grylls19', # 'Grylls19' or 'Moster'
    'BH_mass_method':"Reines&Volonteri15", #"Shankar16", "KormendyHo", "Eq4", "Davis18", "Sahu19" and "Reines&Volonteri15"
    'BH_mass_scatter':"Intrinsic", # "Intrinsic" or float
    'duty_cycle':"Schulze", # "Schulze", "Man16", "Geo" or float (0.18)
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
parameters = [-3.0,-2.5,-2.0,-1.75,-1.5]

################################
# Edd ratio parameters definition:
sigma_z=0.3
mu_z=-2.25

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

norm=9.5
suffix=f'_norm{norm}'