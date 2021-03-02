# set pars for z=1, same schechter as standard, scaling relation of Davis with original slope but extend mass range

import numpy as np

################################
# set simulation parameters
z = 1.0
reds_dic={0.45:0, 1:1, 1.7:2, 2.7:3}
index=reds_dic.get(z) # needed for IDL data

methods={'halo_to_stars':'Grylls19', # 'Grylls19' or 'Moster'
    'BH_mass_method':"Davis18", #"Shankar16", "KormendyHo", "Eq4", "Davis18", "Sahu19" and "Reines&Volonteri15"
    'BH_mass_scatter':"Intrinsic", # "Intrinsic" or float
    'duty_cycle':"Schulze", # "Schulze", "Man16", "Geo" or float (0.18)
    'edd_ratio':"Schechter", # "Schechter", "PowerLaw", "Gaussian", "Geo"
    'bol_corr':'Lusso12_modif', # 'Duras20', 'Marconi04', 'Lusso12_modif'
    'SFR':'Carraro20' # 'Tomczak16', "Schreiber15", "Carraro20"
    }

################################
# Edd ratio parameters definition:
#fitting:
#redshift = [0.1, 1, 2]
#alpha = [-0.25,1.6,7.14]
#lambd = [0.05, -0.8,-0.5]
#alpha_pars=np.polyfit(redshift,alpha,2)
#lambda_pars=np.polyfit(redshift,lambd,2)
#np.savez('schechter_pars.npz',alpha_pars=alpha_pars,lambda_pars=lambda_pars)

schechter_pars=np.load('schechter_pars.npz')
alpha_pars=schechter_pars['alpha_pars']
lambda_pars=schechter_pars['lambda_pars']

alpha_pol=np.poly1d(alpha_pars)
lambda_pol=np.poly1d(lambda_pars)

alpha_z=alpha_pol(z)
lambda_z=lambda_pol(z)

alpha_z=0.15
lambda_z=0.1


################################
# mass range restrictions
M_inf=0
M_sup=0

################################
# filename suffix
suffix='_slope3.05'
