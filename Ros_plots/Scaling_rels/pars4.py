# set pars for z=1, same schechter as standard, change scaling relation to sahu

import numpy as np

################################
# set simulation parameters
z = 0.45
reds_dic={0.45:0, 1:1, 1.7:2, 2.7:3}
index=reds_dic.get(z) # needed for IDL data

methods={'halo_to_stars':'Grylls19', # 'Grylls19' or 'Moster'
    'BH_mass_method':"Sahu19", #"Shankar16", "KormendyHo", "Eq4", "Davis18", "Sahu19" and "Reines&Volonteri15"
    'BH_mass_scatter':"Intrinsic", # "Intrinsic" or float
    'duty_cycle':"Schulze", # "Schulze", "Man16", "Geo" or float (0.18)
    'edd_ratio':"Gaussian", # "Schechter", "PowerLaw", "Gaussian", "Geo"
    'bol_corr':'Lusso12_modif', # 'Duras20', 'Marconi04', 'Lusso12_modif'
    'SFR':'Carraro20' # 'Tomczak16', "Schreiber15", "Carraro20"
    }

################################
# Edd ratio parameters definition:
sigma_z=0.3
mu_z=-2.25

if methods['edd_ratio']=='Gaussian':
   lambda_z=sigma_z
   alpha_z=mu_z

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
    M_inf=10.
print(M_inf,M_sup)

################################
# filename suffix
suffix='Sahu19'
