#%%
"""
Plots
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.io import readsav
from Plots import get_cycle, read_dfs, comp_subplot, comp_plot

curr_dir=os.getcwd()
#%%
# import data from IDL

read_data = readsav('vars_EuclidAGN_90.sav',verbose=True)

data={}
for key, val in read_data.items():
    data[key]=np.copy(val)
    data[key][data[key] == 0.] = np.nan
#print(data.keys())
#%%
# Universe parameters
z = 1.
reds_dic={0.45:0, 1:1, 1.7:2, 2.7:3}
i=reds_dic.get(z) # needed for IDL data

methods={'halo_to_stars':'Grylls et al. (2019)', # 'Grylls19' or 'Moster'
    'BH_mass_method':"Shankar et al. (2016)", #"Shankar16", "KormendyHo", "Eq4", "Davis18", "Sahu19" and "Reines&Volonteri15"
    'BH_mass_scatter':"Intrinsic", # "Intrinsic" or float
    'duty_cycle':"Schulze et al (2015)", # "Schulze", "Man16", "Geo" or float (0.18)
    'edd_ratio':"Schechter", # "Schechter", "PowerLaw", "Gaussian", "Geo"
    'bol_corr':'modified Lusso et al. (2012)', # 'Duras20', 'Marconi04', 'Lusso12_modif'
    'SFR':'Carraro20' # 'Tomczak16', "Schreiber15", "Carraro20"
    }

#%%
#Plot parameters
# Generic definitions

# plot global properties
# Valid font sizes are xx-small, x-small, small, medium, large, x-large, xx-large, smaller, larger.
params = {'legend.fontsize': 'large',
          'legend.title_fontsize':'large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'lines.markersize' : 8,
         'xtick.labelsize':'large',
         'ytick.labelsize':'large',
         'xtick.top': True,
         'xtick.direction':'in',
         'ytick.right': True,
         'ytick.direction': 'in',
         'xtick.minor.visible':True}
plt.rcParams.update(params)
text_pars=dict(horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor='gray', alpha=0.5))
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
ls=['--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10)),'--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]
cols = plt.cm.tab10.colors
markers = ["o","^","p","P","*","h","X","D","8"]

# change color map
from matplotlib.pyplot import cycler
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm
import re

#%%
###############################
######### Figs 2 ###############
###############################
# plot global properties
plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")
params = {'legend.fontsize': 'small',
          'legend.title_fontsize':'small'}
plt.rcParams.update(params)


# make 4 comparison plots - all separated
##########################################################
# top-right
# # Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}*_sigma0.3.csv')
paths = sorted(paths)
print(paths)
#read DFs
df_dic = read_dfs(paths)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nDuty Cycle: {methods['duty_cycle']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
leg_title='Eddington ratio distribution'
comp_plot(df_dic,filename='Fig2_tr.pdf',leg_title=leg_title,i=i)
##########################################################
# bottom-right
# Comparison of Davis slopes
# define DF path
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Davis_slope/bs_perc_z{z}*.csv'))
paths = sorted(paths, reverse=True)
print(paths)
keys=['Davis et al. (2018) extended',r'Slope $\beta=2.5$',r'Slope $\beta=2.0$',r'Slope $\beta=1.5$',r'Slope $\beta=1.0$']
#read DFs
df_dic = read_dfs(paths,keys)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}"
leg_title='Scaling relation\nwith varying slope'
comp_plot(df_dic,filename='Fig2_br.pdf',leg_title=leg_title,i=i)

##########################################################
# bottom-left
# Duty Cycle comparison
paths = glob.glob(curr_dir+f'/Ros_plots/Standard/bs_perc_z{z}*.csv') + glob.glob(curr_dir+f'/Ros_plots/Duty_Cycles/bs_perc_z{z}_lambda-0.80_alpha1.60*.csv')
#paths = sorted(paths)
print(paths)
keys=['Schulze et al. (2015)','Georgakakis et al. (2017)','Man et al. (2019)','const=0.2']
#read DFs
df_dic = read_dfs(paths,keys)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
leg_title=r'Duty cycle method'
comp_plot(df_dic,filename='Fig2_bl.pdf',leg_title=leg_title,i=i)
##########################################################
"""
# common lables:
fig.text(0.5, 0.08, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.08, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
plt.ylim(2.1e-6,9e2)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/fig2_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
"""

#%%
############################
##### Fig 3 ######
############################
# plot global properties
plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")
params = {'legend.fontsize': 'medium',
          'legend.title_fontsize':'medium'}
plt.rcParams.update(params)

reds_dic={0.45:0, 1:1, 1.7:2, 2.7:3}
# make 1x2 comparison plots - reduced version with R&V  only
fig,axs = plt.subplots(1,2,figsize=[15, 6], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

##########################################################
# bottom
methods['BH_mass_method']="Reines & Volonteri (2015)"

##########################################################
# left
z = 0.45
i=reds_dic.get(z)
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths)

leg_title=f"z = {z:.1f}\nBH_mass: {methods['BH_mass_method']}"
comp_subplot(axs[0],df_dic,leg_title=leg_title,m_min=4,i=i)

##########################################################
# right
z = 2.7
i=reds_dic.get(z)
# # Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths)
print('i=',i)

leg_title=f"z = {z:.1f}\nBH_mass: {methods['BH_mass_method']}"
comp_subplot(axs[1],df_dic,leg_title=leg_title,m_min=4,i=i)
##########################################################
# common lables:
fig.text(0.5, 0.05, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.06, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
plt.ylim(5e-4,1.8e2)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/fig3_conference.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;

# go back to previous z:
z = 1.
i=reds_dic.get(z)
methods['BH_mass_method']="Shankar et al. (2016)"
