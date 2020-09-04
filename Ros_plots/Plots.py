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

curr_dir=os.getcwd()
#%%
# import data from IDL

read_data = readsav('../vars_EuclidAGN_90.sav',verbose=True)

data={}
for key, val in read_data.items():
    data[key]=np.copy(val)
    data[key][data[key] == 0.] = np.nan
print(data.keys())
#%%
# Universe parameters
z = 1.
reds_dic={0.45:0, 1:1, 1.7:2, 2.7:3}
index=reds_dic.get(z) # needed for IDL data

methods={'halo_to_stars':'Grylls19', # 'Grylls19' or 'Moster'
    'BH_mass_method':"Shankar16", #"Shankar16", "KormendyHo", "Eq4", "Davis18", "Sahu19" and "Reines&Volonteri15"
    'BH_mass_scatter':"Intrinsic", # "Intrinsic" or float
    'duty_cycle':"Schulze", # "Schulze", "Man16", "Geo" or float (0.18)
    'edd_ratio':"Schechter", # "Schechter", "PowerLaw", "Gaussian", "Geo"
    'bol_corr':'Lusso12_modif', # 'Duras20', 'Marconi04', 'Lusso12_modif'
    'SFR':'Carraro20' # 'Tomczak16', "Schreiber15", "Carraro20"
    }

#%%
#Plot parameters
# Generic definitions
i=index

# plot global properties
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
         'ytick.direction': 'in'}
plt.rcParams.update(params)
text_pars=dict(horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor='gray', alpha=0.5))
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
ls=['--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]
cols = plt.cm.tab10.colors
markers = ["o","^","p","P","*","h","X","D","8"]

# change color map
from matplotlib.pyplot import cycler
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm

def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return cycler("color",colors)

# read files as dataframes
def read_dfs(keys,paths):
    #read and place in dictionary
    dfs=[pd.read_csv(p,header=[0,1],index_col=0) for p in paths]
    # percentile values in column names to float type instead of string
    for df in dfs:
        df.columns.set_levels(df.columns.levels[1].astype(float),level=1,inplace=True)

    return dict(zip(keys, dfs))

# make comparison plot
def comp_plot(df_dic,method_legend,filename='Comparisons',leg_title=None):
    fig,ax = plt.subplots(figsize=[9, 6])
    #plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")

    # "real" datapoints
    ax.scatter(data['m_ave'][0,2:,i], data['l_ave'][0,2:,i], edgecolors='Black', marker="s",label='Carraro et al. (2020)')
    ax.errorbar(data['m_ave'][0,2:,i], data['l_ave'][0,2:,i],
                    yerr=np.array([data['l_ave'][0,2:,i] - data['l_ave'][2,2:,i], 
                        data['l_ave'][1,2:,i] - data['l_ave'][0,2:,i]]),
                    linestyle='solid', zorder=0)

    # simulated datasets
    for j,(s,df) in enumerate(df_dic.items()):
         # errorbars of bootstrapped simulation points
         xerr=np.array([df['SFR',0.5] - df['SFR',0.05], 
                     df['SFR',0.95] - df['SFR',0.5]])
         yerr=np.array([df['luminosity',0.5] - df['luminosity',0.05], 
                        df['luminosity',0.95] - df['luminosity',0.5]])

         ax.scatter(df['stellar_mass',0.5],df['luminosity',0.5], edgecolors='Black', label=s)
         ax.errorbar(df['stellar_mass',0.5],df['luminosity',0.5], 
                           yerr=yerr, linestyle=ls[j], zorder=0)

    #plt.text(0.83, 0.41, f'z = {z:.1f}', transform=fig.transFigure, **text_pars)
    plt.text(0.137, 0.865, f'z = {z:.1f}\n'+method_legend, transform=fig.transFigure, **text_pars)
    #ax.set_ylim(1.5e-2,7e1)
    ax.set_yscale('log')
    ax.set_xlabel('M$_*$ (M$_\odot$)')
    ax.set_ylabel('L$_X$ (2-10 keV) / $10^{42}$ (erg/s)')
    ax.legend(loc='lower right',title=leg_title)
    plt.savefig(curr_dir+'/'+filename+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
    return


#%%
# Plot SFR vs LX
def SFR_LX(df_dic,data,leg_title=None):
   handles=[]

   _min = np.nanmin(data['m_ave'][0,2:,:])
   _max = np.nanmax(data['m_ave'][0,2:,:])
   for s,bs_perc in df_dic.items():
      # define parameters for datapoints' colors and colorbar
      _min = np.minimum(np.nanmin(bs_perc['stellar_mass',0.5]),_min)
      _max = np.maximum(np.nanmax(bs_perc['stellar_mass',0.5]),_max)

   fig = plt.figure(figsize=[12, 8])
   #plt.rcParams['figure.figsize'] = [12, 8]

   for j,(s,bs_perc) in enumerate(df_dic.items()):
      # errorbars of bootstrapped simulation points
      xerr=np.array([bs_perc['SFR',0.5] - bs_perc['SFR',0.05], 
                  bs_perc['SFR',0.95] - bs_perc['SFR',0.5]])
      yerr=np.array([bs_perc['luminosity',0.5] - bs_perc['luminosity',0.05], 
                     bs_perc['luminosity',0.95] - bs_perc['luminosity',0.5]])

      # simulated datapoints
      plt.scatter(bs_perc['SFR',0.5],bs_perc['luminosity',0.5], vmin = _min, vmax = _max, marker=markers[j], edgecolors='Black',
                  c=bs_perc['stellar_mass',0.5] , s=bs_perc['stellar_mass',0.5]*10)
      plt.errorbar(bs_perc['SFR',0.5],bs_perc['luminosity',0.5],
                     xerr=xerr, yerr=yerr, linestyle=ls[j], c=cols[j+1], zorder=0)
      handles += [mlines.Line2D([], [], color=cols[j+1], linestyle=ls[j], 
                                 marker=markers[j],markeredgecolor='Black',
                                 label=s)]

   # "real" datapoints
   data_pars=dict(marker="s",linestyle='-')
   sc=plt.scatter(data['sfr_ave'][0,2:,i], data['l_ave'][0,2:,i], vmin = _min, vmax = _max, edgecolors='Black',
               c=data['m_ave'][0,2:,0], s=data['m_ave'][0,2:,0]*10, marker=data_pars['marker'])
   plt.errorbar(data['sfr_ave'][0,2:,i], data['l_ave'][0,2:,i],
                  xerr=[data['sfr_ave'][0,2:,i]-data['sfr_ave'][2,2:,i],
                     data['sfr_ave'][1,2:,i]-data['sfr_ave'][0,2:,i]],
                  yerr=np.array([data['l_ave'][0,2:,i] - data['l_ave'][2,2:,i], 
                     data['l_ave'][1,2:,i] - data['l_ave'][0,2:,i]]),
                  linestyle=data_pars['linestyle'], zorder=0, color=cols[0])
   handles += [mlines.Line2D([], [], color=cols[0], markeredgecolor='Black',
                              label='Carraro et al. (2020)',**data_pars)]

   # colorbar, labels, legend, etc
   plt.colorbar(sc).set_label('Stellar mass (M$_\odot$)')
   #plt.text(0.137, 0.78, f'z = {z:.1f}', transform=fig.transFigure, **text_pars)
   plt.text(0.137, 0.865, f"z = {z:.1f}\nHalo to M* = {methods['halo_to_stars']}\nEddington ratio = {methods['edd_ratio']}\nDuty Cycle = {methods['duty_cycle']}\nBH_mass = {methods['BH_mass_method']}\nBol corr = {methods['bol_corr']}", transform=fig.transFigure, **text_pars)
   plt.xscale('log')
   plt.yscale('log')
   plt.xlabel('SFR (M$_\odot$/yr)')
   plt.ylabel('L$_X$ (2-10 keV) / $10^{42}$ (erg/s)')
   plt.legend(handles=handles, loc='lower right',title=leg_title)
   plt.savefig(curr_dir+f'/SFvsLX_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 


#%%
# Comparison of scaling relations
# define DF path
paths = glob.glob(curr_dir+f'/*Reines*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/*Davis*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/*Sahu_extended*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/03*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Reines & Volonteri (2015)', 'Davis et al. (2018)', 'Sahu et al. (2019)', 'Shankar et al. (2016)']
#read DFs
df_dic = read_dfs(keys,paths)
leg_title='Scaling relation'

SFR_LX(df_dic,data,leg_title)


#%%
# Comparison of K&H slopes
# define DF path
paths = glob.glob(curr_dir+f'/1*Kormendy*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=[p[p.rfind('/')-10:p.rfind('/')] for p in paths]
#read DFs
df_dic = read_dfs(keys,paths)

method_legend=f"Halo to M* = {methods['halo_to_stars']}\nEddington ratio = {methods['edd_ratio']}\nDuty Cycle = {methods['duty_cycle']}\nBol corr = {methods['bol_corr']}"
filename='Comp_K&H_slopes'
leg_title='Kormendy & Ho slopes'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)



#%%
# Comparison of BH_mass_method: K&H, shankar, Eq4
# define DF path
paths = glob.glob(curr_dir+f'/17*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/14*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/03*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Shankar et al. (2016)','Kormendy & Ho (2013)','Eq. 4']
#read DFs
df_dic = read_dfs(keys,paths)

method_legend=f"Halo to M* = {methods['halo_to_stars']}\nEddington ratio = {methods['edd_ratio']}\nDuty Cycle = {methods['duty_cycle']}\nBol corr = {methods['bol_corr']}"
filename='Comp_BH_mass_method'
leg_title=r'M$_* - {\rm M}_{\rm BH}$ method'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)


#%%
# halo_to_stars comparison
paths = glob.glob(curr_dir+f'/03*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/06*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Gryllis et al. (2019)','Moster et al. (----)']
#read DFs
df_dic = read_dfs(keys,paths)

method_legend=f"Eddington ratio = {methods['edd_ratio']}\nDuty Cycle = {methods['duty_cycle']}\nBH_mass = {methods['BH_mass_method']}\nBol corr = {methods['bol_corr']}"
filename='Comp_halo_to_stars'
leg_title=r'M$_{\rm halo}-{\rm M}_*$ method'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)


#%%
# halo_to_stars comparison
paths = glob.glob(curr_dir+f'/03*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/*DutyCycle*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Schulze et al. (2016)','const=0.18','Man et al. (2016)','Geo et al. (----)']
#read DFs
df_dic = read_dfs(keys,paths)

method_legend=f"Halo to M* = {methods['halo_to_stars']}\nEddington ratio = {methods['edd_ratio']}\nBH_mass = {methods['BH_mass_method']}\nBol corr = {methods['bol_corr']}"
filename='Comp_duty_cycles'
leg_title=r'Duty cycle method'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)


#%%
# Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/03*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/*Gaussian*m=0.25*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Schechter',r'Gaussian $\mu=0.25$, $\sigma=0.05$',r'Gaussian $\mu=0.25$, $\sigma=0.20$',r'Gaussian $\mu=0.25$, $\sigma=0.40$']
#read DFs
df_dic = read_dfs(keys,paths)

method_legend=f"Halo to M* = {methods['halo_to_stars']}\nDuty Cycle = {methods['duty_cycle']}\nBH_mass = {methods['BH_mass_method']}\nBol corr = {methods['bol_corr']}"
filename='Comp_Edd_gaussian_width'
leg_title='Gaussian Eddington ratio\nwith varying width'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)


#%%
# Gaussian mean Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/03*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/*Gaussian*s=0.2*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Schechter',r'Gaussian $\mu=0.05$, $\sigma=0.20$',r'Gaussian $\mu=0.25$, $\sigma=0.20$',r'Gaussian $\mu=0.60$, $\sigma=0.20$']
#read DFs
df_dic = read_dfs(keys,paths)

method_legend=f"Halo to M* = {methods['halo_to_stars']}\nDuty Cycle = {methods['duty_cycle']}\nBH_mass = {methods['BH_mass_method']}\nBol corr = {methods['bol_corr']}"
filename='Comp_Edd_gaussian_mean'
leg_title='Gaussian Eddington ratio\nwith varying mean'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)


#%%



#%%