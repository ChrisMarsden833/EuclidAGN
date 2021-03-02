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
# SF
read_data = readsav('./IDL_data/vars_EuclidAGN_90.sav')#,verbose=True

data={}
for key, val in read_data.items():
    data[key]=np.copy(val)
    data[key][data[key] == 0.] = np.nan
#print(data.keys())

# Q
read_data = readsav('./IDL_data/vars_EuclidAGN_90_Q.sav')#,verbose=True
data_Q={}
for key, val in read_data.items():
    data_Q[key]=np.copy(val)
    data_Q[key][data_Q[key] == 0.] = np.nan

# SB
read_data = readsav('./IDL_data/vars_EuclidAGN_90_SB.sav')#,verbose=True

data_SB={}
for key, val in read_data.items():
    data_SB[key]=np.copy(val)
    data_SB[key][data_SB[key] == 0.] = np.nan
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
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'lines.markersize' : 8,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'xtick.top': True,
         'xtick.direction':'in',
         'ytick.right': True,
         'ytick.direction': 'in',
         'xtick.minor.visible':True}
plt.rcParams.update(params)
text_pars=dict(horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor='gray', alpha=0.5))
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
ls=4*['--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10)),'--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]
# possible pallette:
#003f5c,#58508d,#bc5090,#ff6361,#ffa600
#003f5c,#7a5195,#ef5675,#ffa600
#720975,#bf2957,#f07d3a,#d6c449,#98f272
#720975,#cd424c,#eea503,#98f272
#["#a2e4fd","#7bd743","#e46d5e","#00595c","#4b0000"]
#["#92ffff","#7bd743","#e45e60","#005f66","#430000"]
cols = 4*plt.cm.tab10.colors 
markers = 4*["o","^","p","P","*","h","X","D","8"]

# change color map
from matplotlib.pyplot import cycler
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm
import re

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
def read_dfs(paths,keys=None,sigma=True,lambdac=False):
    #read and place in dictionary
    if keys:
      dictionary={}
      for p,key in zip(paths,keys):
         df=pd.read_csv(p,header=[0,1],index_col=0)
         # percentile values in column names to float type instead of string
         df.columns.set_levels(df.columns.levels[1].astype(float),level=1,inplace=True)
         dictionary[key]=df
      return dictionary
    else:
      pattern=re.compile('([a-z]+)(\-*\d*\.\d*)') 
      df_dict={}
      for p in paths:
         df=pd.read_csv(p,header=[0,1],index_col=0)
         df.columns.set_levels(df.columns.levels[1].astype(float),level=1,inplace=True)
         pars=dict(re.findall(pattern, p))
         if '_SB.csv' in p:
            new_key='SB '
         elif '_Q.csv' in p:
            new_key='Q '
         else:
            new_key=''

         if 'Gaussian' in p:
            if 'norm' in pars.keys():
               new_key+=fr"norm={pars['norm']}; "
            new_key+=fr"Gaussian $\mu={pars['mean']}$"
            if sigma:
                  new_key+=fr", $\sigma={pars['sigma']}$"

            #df_dict[fr"Gaussian $\mu={pars['mean']}$"]=df
         else:
            new_key+=fr"Schechter $x*={pars['lambda']}$, $\alpha={pars['alpha']}$"
         if lambdac and ('lambdac' in pars.keys()):
               new_key+=fr"; $\zeta_c={pars['lambdac']}$"
         df_dict[new_key]=df

      return df_dict

#%%
# function for comparison subplots
def comp_subplot(ax,df_dic,method_legend=None,leg_title=None,m_min=2,i=0,Q=False,SB=False,legend=True,legend_loc='lower right'):

    # "real" datapoints
    if Q or SB:
      label='SF Carraro et al. (2020)' 
    else: 
      label='Carraro et al. (2020)'
    subplot_data_LX(ax,data, label, marker='s',m_min=m_min)

    # "real" datapoints
    if Q==True:
      subplot_data_LX(ax,data_Q, label='Q Carraro et al. (2020)', marker="X",m_min=m_min)

    # "real" datapoints
    if SB==True:
      subplot_data_LX(ax,data_SB, label='SB Carraro et al. (2020)', marker="d",m_min=0)

    # simulated datasets
    for j,(s,df) in enumerate(df_dic.items()):
         # errorbars of bootstrapped simulation points
         xerr=np.array([df['SFR',0.5] - df['SFR',0.05], 
                     df['SFR',0.95] - df['SFR',0.5]])
         yerr=np.array([df['luminosity',0.5] - df['luminosity',0.05], 
                        df['luminosity',0.95] - df['luminosity',0.5]])

         ax.scatter(df['stellar_mass',0.5],df['luminosity',0.5], edgecolors='Black', label=s,color=cols[j+1])
         ax.errorbar(df['stellar_mass',0.5],df['luminosity',0.5], 
                           yerr=yerr, linestyle=ls[j], zorder=0,color=cols[j+1])

    #plt.text(0.83, 0.41, f'z = {z:.1f}', transform=fig.transFigure, **text_pars)
    if method_legend:
      ax.text(0.03, 0.965, method_legend, transform=ax.transAxes, **text_pars)
    if legend:
      leg=ax.legend(loc=legend_loc,title=leg_title)
      leg._legend_box.align= "left"
    return

def subplot_data_LX(ax,data, label, marker,m_min=2):
    ax.scatter(data['m_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i], edgecolors='Black', marker=marker,color=cols[0],label=label)
    ax.errorbar(data['m_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i],
                    yerr=np.array([data['l_ave'][0,m_min:,i] - data['l_ave'][2,m_min:,i], 
                        data['l_ave'][1,m_min:,i] - data['l_ave'][0,m_min:,i]]),
                    linestyle='solid',color=cols[0], zorder=0)
    return

#%%
# make 1 single comparison plot
def comp_plot(df_dic,method_legend=None,filename='Comparisons',leg_title=None,i=0,Q=False,SB=False,m_min=2,legend_loc='lower right'):
    fig,ax = plt.subplots(figsize=[9, 6])
    #plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")

    comp_subplot(ax,df_dic,method_legend,leg_title,i=i,Q=Q,SB=SB,m_min=m_min,legend_loc=legend_loc)
    
    ax.set_yscale('log')
    #ax.set_ylim(3e-3,30)
    #ax.set_xlim(9.,11.25)
    ax.set_xlabel('<M$_*$> (M$_\odot$)')
    ax.set_ylabel('<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)')
    plt.savefig(curr_dir+'/Ros_plots/'+filename+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
    return
#%%
# Plot SFR vs LX
#def SFR_LX(df_dic,data,leg_title=None,m_min=2):
def SFR_LX(df_dic,data,leg_title=None,m_min=2,filename='SFvsLX',SB=[],Q=[]):
   fig,ax = plt.subplots(figsize=[12, 8])
   
   SFR_LX_subplot(ax,df_dic,data,leg_title=leg_title,m_min=m_min,SB=SB,Q=Q)

   #plt.text(0.137, 0.78, f'z = {z:.1f}', transform=fig.transFigure, **text_pars)
   #plt.text(0.137, 0.865, f"z = {z:.1f}\nHalo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}", transform=fig.transFigure, **text_pars)
   ax.set_ylabel('<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)')
   plt.savefig(curr_dir+'/Ros_plots/'+filename+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 

def SFR_LX_subplot(ax,df_dic,data,leg_title=None,m_min=2,SB=[],Q=[]):
   handles=[]

   # define parameters for datapoints' colors and colorbar
   _min = np.nanmin(data['m_ave'][0,m_min:,:])
   _max = np.nanmax(data['m_ave'][0,m_min:,:])
   if Q:
      _min = np.minimum(np.nanmin(data_Q['m_ave'][0,m_min:,:]),_min)
      _max = np.maximum(np.nanmax(data_Q['m_ave'][0,m_min:,:]),_max)
   if SB:
      _min = np.minimum(np.nanmin(data_SB['m_ave'][0,:,:]),_min)
      _max = np.maximum(np.nanmax(data_SB['m_ave'][0,:,:]),_max)
   for s,bs_perc in df_dic.items():
      _min = np.minimum(np.nanmin(bs_perc['stellar_mass',0.5]),_min)
      _max = np.maximum(np.nanmax(bs_perc['stellar_mass',0.5]),_max)
   #plt.rcParams['figure.figsize'] = [12, 8]

   for j,(s,bs_perc) in enumerate(df_dic.items()):
      if s in SB:
         SFR_str='SFR_SB'
         s='SB '+s
      elif s in Q:
         SFR_str='SFR_Q'
         s='Q '+s
      else: 
         if Q or SB:
            s='SF '+s
         SFR_str='SFR'

      # errorbars of bootstrapped simulation points
      xerr=np.array([bs_perc[SFR_str,0.5] - bs_perc[SFR_str,0.05], 
                  bs_perc[SFR_str,0.95] - bs_perc[SFR_str,0.5]])
      yerr=np.array([bs_perc['luminosity',0.5] - bs_perc['luminosity',0.05], 
                     bs_perc['luminosity',0.95] - bs_perc['luminosity',0.5]])

      # simulated datapoints
      data_pars=dict(marker=markers[j],linestyle=ls[j],color=cols[j+1], markeredgecolor='Black')
      sc=ax.scatter(bs_perc[SFR_str,0.5],bs_perc['luminosity',0.5], vmin = _min, vmax = _max, marker=data_pars['marker'], edgecolors='Black',
                  c=bs_perc['stellar_mass',0.5] , s=bs_perc['stellar_mass',0.5]*10)
      ax.errorbar(bs_perc[SFR_str,0.5],bs_perc['luminosity',0.5],
                     xerr=xerr, yerr=yerr, linestyle=data_pars['linestyle'], c=data_pars['color'], zorder=0)
      handles += [mlines.Line2D([], [], label=s, **data_pars)]

   # "real" datapoints
   if Q or SB:
      label='SF Carraro et al. (2020)' 
   else: 
      label='Carraro et al. (2020)'
   handles+=subplot_data_SFR(ax,data, label,marker="s",_min=_min,_max=_max,m_min=m_min)

   if Q:
      label='Q Carraro et al. (2020)'
      handles+=subplot_data_SFR(ax,data_Q, label,marker="X",_min=_min,_max=_max,m_min=m_min)

   if SB:
      label='SB Carraro et al. (2020)'
      handles+=subplot_data_SFR(ax,data_SB, label,marker="d",_min=_min,_max=_max,m_min=0)

   # colorbar, labels, legend, etc
   fig.colorbar(sc,pad=0.005).set_label('Stellar mass (M$_\odot$)')
   leg=ax.legend(handles=handles, loc='lower right',title=leg_title,handlelength=3)
   leg._legend_box.align= "left"
   #ax.legend(handles=handles, loc='lower right',title=leg_title,handlelength=3)
   ax.set_xlabel('<SFR> (M$_\odot$/yr)')
   ax.set_xscale('log')
   ax.set_yscale('log')
   return

def subplot_data_SFR(ax,data, label, marker,_min,_max,m_min=2):
   data_pars=dict(marker=marker,linestyle='-', markeredgecolor='Black', color=cols[0])
   ax.scatter(data['sfr_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i], vmin = _min, vmax = _max, edgecolors=data_pars['markeredgecolor'],
               c=data['m_ave'][0,m_min:,0], s=data['m_ave'][0,m_min:,0]*10, marker=data_pars['marker'])
   ax.errorbar(data['sfr_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i],
                  xerr=[data['sfr_ave'][0,m_min:,i]-data['sfr_ave'][2,m_min:,i],
                     data['sfr_ave'][1,m_min:,i]-data['sfr_ave'][0,m_min:,i]],
                  yerr=np.array([data['l_ave'][0,m_min:,i] - data['l_ave'][2,m_min:,i], 
                     data['l_ave'][1,m_min:,i] - data['l_ave'][0,m_min:,i]]),
                  linestyle=data_pars['linestyle'], zorder=0, color=data_pars['color'])
   handle = [mlines.Line2D([], [], label=label,**data_pars)]
   return handle

"""
#%%
###############################
######### Fig 1 ###############
###############################
# Comparison of scaling relations SFR vs LX
# define DF path
paths = glob.glob(curr_dir+f'/Ros_plots/First_version_paper/*Reines*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/Ros_plots/First_version_paper/*Davis*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/Ros_plots/First_version_paper/*Sahu_extended*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/Ros_plots/First_version_paper/03*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Shankar et al. (2016)', 'Reines & Volonteri (2015)', 'Davis et al. (2018)', 'Sahu et al. (2019)']
#read DFs
df_dic = read_dfs(paths,keys)
leg_title='Scaling relation'

SFR_LX(df_dic,data,leg_title)
#%%
###############################
######### Fig 1 ###############
###############################
# Comparison of scaling relations SFR vs LX - CORRECT LENGTH
# define DF path
#paths = glob.glob(curr_dir+f'/Ros_plots/Standard/bs_perc_z{z}*.csv') + sorted(glob.glob(curr_dir+f'/Ros_plots/Scaling_rels_bestLF/bs_perc_*_z{z}*.csv'))
paths = glob.glob(curr_dir+f'/Ros_plots/Standard/bs_perc_z{z}*.csv') + sorted(glob.glob(curr_dir+f'/Ros_plots/Scaling_rels_restr/bs_perc_z{z}*.csv'))
print(paths)
keys=['Shankar et al. (2016)', 'Davis et al. (2018)', 'Reines & Volonteri (2015)', 'Sahu et al. (2019)']
#read DFs
df_dic = read_dfs(paths,keys)
leg_title='Scaling relation'

SFR_LX(df_dic,data,leg_title, m_min = 4)

#%%
"""
###############################
######### Fig 2 ###############
###############################
# plot global properties
plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")
params = {'legend.fontsize': 'small',
          'legend.title_fontsize':'small'}
plt.rcParams.update(params)


# make 4 comparison plots
fig,axs = plt.subplots(2,2,figsize=[15, 10], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

##########################################################
# top-left
paths = [curr_dir+'/Ros_plots/Standard/bs_perc_z1.0_lambda0.10_alpha0.15_lambdac-1.562.csv'] + \
         sorted(glob.glob(curr_dir+f'/Ros_plots/Scaling_rels/bs_perc_z{z}_lambda0.10_alpha0.15*.csv') )
#print(paths)
keys=['Shankar et al. (2016)', 'Davis et al. (2018)', 'Reines & Volonteri (2015)', 'Sahu et al. (2019)']
#read DFs
df_dic = read_dfs(paths,keys)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}"
leg_title=r"M$_{\rm BH}$-M$_*$ scaling relation"
comp_subplot(axs[0,0],df_dic,leg_title=leg_title,i=i)

##########################################################
# top-right
# # Gaussian width Edd ratio distributions comparison
paths = [curr_dir+'/Ros_plots/Standard/bs_perc_z1.0_lambda0.10_alpha0.15_lambdac-1.562.csv'] + \
         glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}*.csv')
paths = sorted(paths)

#read DFs
df_dic = read_dfs(paths,lambdac=True)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nDuty Cycle: {methods['duty_cycle']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
leg_title='Eddington ratio distribution'
comp_subplot(axs[0,1],df_dic,leg_title=leg_title,i=i)
##########################################################
# bottom-right
# Comparison of Davis slopes
# define DF path
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Davis_slope/bs_perc_z{z}_lambda0.10_alpha0.15*.csv'))
paths = sorted(paths, reverse=True)

keys=['Davis et al. (2018) extended',r'Slope $\beta=2.5$',r'Slope $\beta=2.0$',r'Slope $\beta=1.5$',r'Slope $\beta=1.0$']
#read DFs
df_dic = read_dfs(paths,keys)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}"
leg_title=r"M$_{\rm BH}$-M$_*$ scaling relation"+"\nwith varying slope"
comp_subplot(axs[1,0],df_dic,leg_title=leg_title,i=i)

##########################################################
# bottom-left
# Duty Cycle comparison
paths = [curr_dir+'/Ros_plots/Standard/bs_perc_z1.0_lambda0.10_alpha0.15_lambdac-1.562.csv'] + \
         glob.glob(curr_dir+f'/Ros_plots/Duty_Cycles/bs_perc_z{z}_lambda0.10_alpha0.15*.csv')
#paths = sorted(paths)

keys=['Schulze et al. (2015)','Georgakakis et al. (2017)','Man et al. (2019)','const=0.2']
#read DFs
df_dic = read_dfs(paths,keys)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
leg_title=r'Duty cycle'
comp_subplot(axs[1,1],df_dic,leg_title=leg_title,i=i)
##########################################################
# invisible labels for creating space:
#axs[-1, 0].set_xlabel('.', color=(0, 0, 0, 0))
#axs[-1, 0].set_ylabel('.', color=(0, 0, 0, 0))
# common lables:
fig.text(0.5, 0.07, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.08, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
plt.ylim(2.1e-6,9e2)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/fig2_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;


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
# make 2x3 comparison plots
#fig,axs = plt.subplots(2,3,figsize=[15, 10], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
# make 1x3 comparison plots
fig,axs = plt.subplots(1,3,figsize=[15, 5], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
sigma=False # don't show the standard deviation of the gaussian in the legend of the plots

"""
##########################################################
# top
methods['BH_mass_method']="Sahu et al. (2019)"
##########################################################
# left
z = 0.45
i=reds_dic.get(z)
paths = glob.glob(curr_dir+f'/Ros_plots/Sahu_Gaussian/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nDuty Cycle: {methods['duty_cycle']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
leg_title=f"z = {z:.2f}"+"\n"+rf"M$_{{\rm BH}}$-M$_*$: {methods['BH_mass_method']}"
comp_subplot(axs[0,0],df_dic,leg_title=leg_title,m_min=4,i=i)

##########################################################
# center
z = 1.0
i=reds_dic.get(z)
paths = glob.glob(curr_dir+f'/Ros_plots/Sahu_Gaussian/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma)

leg_title=f"z = {z:.2f}"+"\n"+rf"M$_{{\rm BH}}$-M$_*$: {methods['BH_mass_method']}"
comp_subplot(axs[0,1],df_dic,leg_title=leg_title,m_min=4,i=i)

##########################################################
# right
z = 2.7
i=reds_dic.get(z)
# # Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/Ros_plots/Sahu_Gaussian/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma)

leg_title=f"z = {z:.2f}"+"\n"+rf"M$_{{\rm BH}}$-M$_*$: {methods['BH_mass_method']}"
comp_subplot(axs[0,2],df_dic,leg_title=leg_title,m_min=4,i=i)
##########################################################
# common lables:
#fig.text(0.5, 0.05, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
#fig.text(0.08, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
#plt.ylim(5e-4,5e3)
#plt.yscale('log')
#plt.savefig(curr_dir+f'/Ros_plots/fig3_sahu.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;

"""
##########################################################
# bottom
methods['BH_mass_method']="Reines & Volonteri (2015)"

##########################################################
# left
z = 0.45
i=reds_dic.get(z)
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}*.csv') 
paths = sorted(paths)
method_legend=rf"M$_{{\rm BH}}$-M$_*$: {methods['BH_mass_method']}"
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma)

#leg_title=f"z = {z:.2f}"+"\n"+rf"M$_{{\rm BH}}$-M$_*$: {methods['BH_mass_method']}"
leg_title=f"z = {z:.2f}"
comp_subplot(axs[0],df_dic,method_legend=method_legend,leg_title=leg_title,m_min=4,i=i)

##########################################################
# center
z = 1.
i=reds_dic.get(z)
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}*.csv') + \
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.0_sigma0.3*.csv.bak') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma)

leg_title=f"z = {z:.2f}"
comp_subplot(axs[1],df_dic,leg_title=leg_title,m_min=4,i=i)

##########################################################
# right
z = 2.7
i=reds_dic.get(z)
# # Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma)

leg_title=f"z = {z:.2f}"
comp_subplot(axs[2],df_dic,leg_title=leg_title,m_min=4,i=i)
##########################################################
# common lables:
fig.text(0.5, 0.04, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.07, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
plt.ylim(5e-4,1.8e2)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/fig3.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;

# go back to previous z:
z = 1.
i=reds_dic.get(z)
methods['BH_mass_method']="Shankar et al. (2016)"
#%%
#####################################################
############ Test lambda char ############
#####################################################
methods['BH_mass_method']="Shankar et al. (2016)"
z = 1.0
i=reds_dic.get(z)
# # Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}_mean-0.1_sigma0.3*.csv*')    + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha1.5_lambda0.0*.csv*')  + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha0.5_lambda1.0*.csv*')  + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}_mean-0.25_sigma0.3*.csv*')   + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha1.0_lambda0.0*.csv*')  + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}_mean-0.5_sigma0.3*.csv*')    + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha1.5_lambda-0.5*.csv*') + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}_mean-0.75_sigma0.3*.csv*')   + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha1.0_lambda-0.5*.csv*') + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha0.5_lambda0.0*.csv*')  + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}_mean-1.0_sigma0.3*.csv*')    + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha1.5_lambda-1.0*.csv*') + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}_mean-1.5_sigma0.3*.csv*')    + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha0.0_lambda1.0*.csv*')  + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}_mean-2.0_sigma0.3*.csv*')    + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha0.0_lambda0.0*.csv*')  + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}_mean-2.5_sigma0.3*.csv*')    + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha0.0_lambda-1.0*.csv*') + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Gaussian/bs_perc_z{z}_mean-3.5_sigma0.3*.csv*')    + \
        glob.glob(curr_dir+f'/Ros_plots/Standard_Schechter/bs_perc_z{z}_alpha-1.0_lambda-1.0*.csv*')
df_dic = read_dfs(paths,lambdac=True)

method_legend=f'z = {z:.2f}\n'+r"M$_{\rm BH}$-M$_*$: "+f"{methods['BH_mass_method']}"
leg_title='Eddington ratio distribution'

comp_plot(df_dic,method_legend,filename='Comparison_lambdachar',leg_title=leg_title,i=i,legend_loc=(1.02,0))

#%%
######################################################################
############ Compare SF,Q,SB to Reines&Volonteri relation ############
######################################################################
# plot global properties
plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")
params = {'legend.fontsize': 'medium',
          'legend.title_fontsize':'medium'}
plt.rcParams.update(params)

method_legend=f'z = {z:.2f}\n'+r"M$_{\rm BH}$-M$_*$: Reines & Volonteri (2015)"
leg_title='Eddington ratio distribution'
zees=[1.0,2.7]
# # Comparison between LX-M* relation of SF,Q,SB galaxies
for z in zees:
   i=reds_dic.get(z)
   paths =glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_*_sigma0.3*.csv') 
   if z==1.0:
      paths+= glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.75_sigma0.3*.csv.bak')
   paths = sorted(paths)
   #read DFs
   df_dic = read_dfs(paths)

   comp_plot(df_dic,method_legend,filename=f'Comparison_SFQSB',leg_title=leg_title,i=i,Q=True,SB=True,m_min=3)

#%%
######################################################################
############ Compare SF,Q,SB LX-SFR relation ############
######################################################################
# Comparison of Eddington ratio distributions SFR vs LX for different galazy types
# define DF path
z = 1.0
i=reds_dic.get(z)
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.5_sigma0.3*.csv') +\
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.75_sigma0.3*.csv.bak') +\
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.5_sigma0.3*.csv') 
#read DFs
df_dic = read_dfs(paths)
#print(df_dic.keys())
leg_title='Eddington ratio distribution'
sb_list=['Gaussian $\\mu=-1.5$, $\\sigma=0.3$']
q_list=['Gaussian $\\mu=-2.5$, $\\sigma=0.3$']

SFR_LX(df_dic,data,leg_title, m_min = 4,filename='SFR_LX_SFQSB',SB=sb_list,Q=q_list)

#%%
######################################################################
############ Compare SF,Q,SB LX-SFR,LX-M* relations together ############
######################################################################
# make 2x3 comparison plots
fig,axs = plt.subplots(1,2,figsize=[15, 5], sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
z = 1.0
i=reds_dic.get(z)

paths =glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_*_sigma0.3*.csv') + \
   glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.75_sigma0.3*.csv.bak')
paths = sorted(paths)
#read DFs
df_dic = read_dfs(paths)

comp_subplot(axs[0],df_dic,method_legend,leg_title=leg_title,i=i,Q=True,SB=True,m_min=3,legend=False)
axs[0].set_xlabel('<M$_*$> (M$_\odot$)')

######################################################################
# second subplot
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.5_sigma0.3*.csv') +\
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.75_sigma0.3*.csv.bak') +\
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.5_sigma0.3*.csv') 
#read DFs
df_dic = read_dfs(paths)
#print(df_dic.keys())
leg_title='Eddington ratio distribution'
sb_list=['Gaussian $\\mu=-1.5$, $\\sigma=0.3$']
q_list=['Gaussian $\\mu=-2.5$, $\\sigma=0.3$']

SFR_LX_subplot(axs[1],df_dic,data,leg_title, m_min = 4,SB=sb_list,Q=q_list)
fig.text(0.07, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
plt.ylim(2e-2,2.5e1)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/fig4.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;


#%%
##########################################################################################
############ Compare SF,Q,SB to Reines&Volonteri relation varying its normalization ############
##########################################################################################
# # Comparison between LX-M* relation of SF,Q,SB galaxies
z = 1.0
i=reds_dic.get(z)
paths =glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.75_sigma0.3*.csv.bak') +glob.glob(curr_dir+f'/Ros_plots/Test_R&V_Gaussian_SFQSB/bs_perc_z{z}_*.csv') 
paths = sorted(paths)
#read DFs
df_dic = read_dfs(paths)

method_legend=f"BH_mass: Reines & Volonteri (2015)"
leg_title='Eddington ratio distribution'

comp_plot(df_dic,method_legend,filename=f'Comparison_SFQSB_z{z}_norm',leg_title=leg_title,i=i,Q=True,SB=True,m_min=3)
#%%
##################################################################################
############ Compare SF,Q,SB LX-SFR relation varying R&V normalization ############
##################################################################################
# Comparison of Eddington ratio distributions SFR vs LX for different galazy types
# define DF path
z = 1.0
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.75_sigma0.3*.csv.bak') +\
        [ curr_dir+f'/Ros_plots/Test_R&V_Gaussian_SFQSB/bs_perc_z{z}_mean-1.75_sigma0.30_norm6.75.csv',
         curr_dir+f'/Ros_plots/Test_R&V_Gaussian_SFQSB/bs_perc_z{z}_mean-1.75_sigma0.30_norm7.75.csv']
#read DFs
df_dic = read_dfs(paths)

leg_title='Eddington ratio distribution'
sb_list=['norm=7.75; Gaussian $\\mu=-1.75$, $\\sigma=0.30$']
q_list=['norm=6.75; Gaussian $\\mu=-1.75$, $\\sigma=0.30$']

SFR_LX(df_dic,data,leg_title, m_min = 4,filename='SFR_LX_SFQSB_norm',SB=sb_list,Q=q_list)
#%%
#####################################################
############ Test scatter Mh-M* ############
#####################################################
# # Comparison between Mh-M* relation with different additional scatter as error
z = 1.0
i=reds_dic.get(z)
paths = [curr_dir+f'/Ros_plots/02_First_draft_circulation/Standard/bs_perc_z{z}_lambda-0.80_alpha1.60.csv', 
         curr_dir+f'/Ros_plots/Standard/bs_perc_z{z}_lambda-0.80_alpha1.60.csv', 
         curr_dir+f'/Ros_plots/Standard/Scatter_Mh-M*_0.3/bs_perc_z{z}_lambda-0.80_alpha1.60.csv', 
         curr_dir+f'/Ros_plots/Standard/Scatter_Mh-M*_0.6/bs_perc_z{z}_lambda-0.80_alpha1.60.csv']
#read DFs
keys=['No scatter','Scatter 0.15 dex','Scatter 0.30 dex','Scatter 0.60 dex']
df_dic = read_dfs(paths,keys)

method_legend=f"BH_mass: Shankar+16"
leg_title='Scatter in MH-M* relation'

comp_plot(df_dic,method_legend,filename='Comparison_Scatter_Mh-M*',leg_title=leg_title,i=i)

#%%
#####################################################
############ Test scatter Mbh-M* ############
#####################################################
# # Comparison between standard configuration and Shankar relation with intrinsic scatter
# or double or quadruple scatter
z = 1.0
i=reds_dic.get(z)
paths = [curr_dir+f'/Ros_plots/Standard/bs_perc_z{z}_lambda-0.80_alpha1.60.csv', 
         curr_dir+f'/Ros_plots/Standard/Scatter_Mbh-M*_2*/bs_perc_z{z}_lambda-0.80_alpha1.60.csv', 
         curr_dir+f'/Ros_plots/Standard/Scatter_Mbh-M*_4*/bs_perc_z{z}_lambda-0.80_alpha1.60.csv']
#read DFs
keys=['Intrinsic','2*Intrinsic','4*Intrinsic']
df_dic = read_dfs(paths,keys)

method_legend=f"BH_mass: Shankar+16"
leg_title='Scatter in Mbh-M* relation'

comp_plot(df_dic,method_legend,filename='Comparison_Scatter_Mbh-M*',leg_title=leg_title,i=i)

"""
#%%
#####################################################
############ OLD ############
#####################################################
# Comparison of K&H slopes
# define DF path
paths = glob.glob(curr_dir+f'/Ros_plots/1*Kormendy*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=[p[p.rfind('/')-10:p.rfind('/')] for p in paths]
#read DFs
df_dic = read_dfs(paths,keys)

method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}"
filename='Comp_K&H_slopes'
leg_title='Kormendy & Ho slopes'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)

#%%
# Comparison of BH_mass_method: K&H, shankar, Eq4
# define DF path
paths = glob.glob(curr_dir+f'/Ros_plots/17*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/Ros_plots/14*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/Ros_plots/03*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Shankar et al. (2016)','Kormendy & Ho (2013)','Eq. 4']
#read DFs
df_dic = read_dfs(paths,keys)

method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}"
filename='Comp_BH_mass_method'
leg_title=r'M$_* - {\rm M}_{\rm BH}$ method'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)

#%%
# halo_to_stars comparison
paths = glob.glob(curr_dir+f'/Ros_plots/03*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/Ros_plots/06*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Gryllis et al. (2019)','Moster et al. (----)']
#read DFs
df_dic = read_dfs(paths,keys)

method_legend=f"Eddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
filename='Comp_halo_to_stars'
leg_title=r'M$_{\rm halo}-{\rm M}_*$ method'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)

#%%
# Duty Cycle comparison
paths = glob.glob(curr_dir+f'/Ros_plots/03*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/Ros_plots/*DutyCycle*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Schulze et al. (2015)','const=0.18','Man et al. (2019)','Georgakakis et al. (2017)']
#read DFs
df_dic = read_dfs(paths,keys)

method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
filename='Comp_duty_cycles'
leg_title=r'Duty cycle method'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)

#%%
# Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/Ros_plots/03*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/Ros_plots/*Gaussian*m=0.25*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Schechter',r'Gaussian $\mu=0.25$, $\sigma=0.05$',r'Gaussian $\mu=0.25$, $\sigma=0.20$',r'Gaussian $\mu=0.25$, $\sigma=0.40$']
#read DFs
df_dic = read_dfs(paths,keys)

method_legend=f"Halo to M*: {methods['halo_to_stars']}\nDuty Cycle: {methods['duty_cycle']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
filename='Comp_Edd_gaussian_width'
leg_title='Gaussian Eddington ratio\nwith varying width'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)

#%%
# Gaussian mean Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/Ros_plots/03*/bs_perc_z{z}.csv') + glob.glob(curr_dir+f'/Ros_plots/*Gaussian*s=0.2*/bs_perc_z{z}.csv')
paths = sorted(paths)
print(paths)
keys=['Schechter',r'Gaussian $\mu=0.05$, $\sigma=0.20$',r'Gaussian $\mu=0.25$, $\sigma=0.20$',r'Gaussian $\mu=0.60$, $\sigma=0.20$']
#read DFs
df_dic = read_dfs(paths,keys)

method_legend=f"Halo to M*: {methods['halo_to_stars']}\nDuty Cycle: {methods['duty_cycle']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
filename='Comp_Edd_gaussian_mean'
leg_title='Gaussian Eddington ratio\nwith varying mean'
comp_plot(df_dic,method_legend,filename,leg_title=leg_title)

"""