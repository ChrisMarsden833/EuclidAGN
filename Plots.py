#%%
# Warning: Using matplotlib.pyplot.plot() determines the width of points. Using matplotlib.pyplot.scatter() determines the area of points.
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
from difflib import SequenceMatcher
from adjustText import adjust_text

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
ls=50*['--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10)),'--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]
# possible pallette:
#003f5c,#58508d,#bc5090,#ff6361,#ffa600
#003f5c,#7a5195,#ef5675,#ffa600
#720975,#bf2957,#f07d3a,#d6c449,#98f272
#720975,#cd424c,#eea503,#98f272
#["#a2e4fd","#7bd743","#e46d5e","#00595c","#4b0000"]
#["#92ffff","#7bd743","#e45e60","#005f66","#430000"]
#cols = 4*plt.cm.tab10.colors 
cols= 50*["#1f77b4","#fda000","#e55171","#44aa99","#332288","#e6a1eb","#0b672a","#f6d740","#bb2aa7","#6ab1d1","#5e0101","#3eba00"]
markers = 50*["o","^","p","P","*","h","X","D","8"]

# change color map
from matplotlib.pyplot import cycler
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

def convert_index_to_float(index):
   new_index=[]
   for idx in index:
      try:
         new_index.append(float(idx))
      except ValueError:
         new_index.append(idx)
   return new_index

# read files as dataframes
def read_dfs(paths,keys=None,sigma=True,mean=True,lambdac=False):
    #read and place in dictionary
    if keys:
      dictionary={}
      for p,key in zip(paths,keys):
         df=pd.read_csv(p,header=[0,1],index_col=0)
         # percentile values in column names to float type instead of string
         #df.columns.set_levels(df.columns.levels[1].astype(float,errors='ignore'),level=1,inplace=True)
         df.columns.set_levels(convert_index_to_float(df.columns.levels[1]),level=1,inplace=True)
         dictionary[key]=df
      return dictionary
    else:
      pattern=re.compile('([a-z]+)(\-*\d*\.\d*)') 
      df_dict={}
      for p in paths:
         df=pd.read_csv(p,header=[0,1],index_col=0)
         #df.columns.set_levels(df.columns.levels[1].astype(float,errors='ignore'),level=1,inplace=True)
         df.columns.set_levels(convert_index_to_float(df.columns.levels[1]),level=1,inplace=True)
         pars=dict(re.findall(pattern, p))
         if '_SB.csv' in p:
            new_key='SB '
         elif '_Q.csv' in p:
            new_key='Q '
         else:
            new_key=''

         if 'Gaussian' in p:
            new_key+=fr"Gaussian"
            if mean:
               new_key+=fr" $\mu={float(pars['mean']):.2f}$"
            if sigma:
                  new_key+=fr", $\sigma={float(pars['sigma']):.2f}$"
            if 'norm' in pars.keys():
               new_key+=fr"; norm={pars['norm']}"

            #df_dict[fr"Gaussian $\mu={pars['mean']}$"]=df
         else:
            new_key+=fr"Schechter $x*={float(pars['lambda']):.2f}$, $\alpha={float(pars['alpha']):.2f}$"
         if (lambdac==True) and ('lambdac' in pars.keys()):
               new_key+=fr"; $\zeta_c={np.asarray(pars['lambdac'], dtype=np.float64):.2f}$"
               #new_key+=fr"; $\zeta_c={pars['lambdac']:.2f}$"
         if 'lambdam' in pars.keys():
               new_key+=r"; $\log(\lambda_{min})=$"
               new_key+=f"{np.asarray(pars['lambdam'], dtype=np.float64):.0f}"
         df_dict[new_key]=df

      return df_dict

#%%
# function for comparison subplots
def comp_subplot(ax,df_dic,method_legend=None,leg_title=None,m_min=2,i=0,
                  SB=[],Q=[],SF=[],legend=True,legend_loc='lower right',markers_style=None,lambda_ave=False,ncol=1):
   #ax.set_yscale('log')
   # "real" datapoints
   if Q or SB:
      label='SF Carraro et al. (2020)' 
   else: 
      label='Carraro et al. (2020)'
   subplot_data_LX(ax,data, label, marker='s',m_min=m_min)

   # "real" datapoints
   if Q:
      subplot_data_LX(ax,data_Q, label='Q Carraro et al. (2020)', marker="X",m_min=4)

   # "real" datapoints
   if SB:
      subplot_data_LX(ax,data_SB, label='SB Carraro et al. (2020)', marker="d",m_min=0)

   # simulated datasets
   markers_size=(plt.rcParams['lines.markersize']*1.2)**2
   edge_width=0.5
   j=-1
   for s,df in df_dic.items():
      if s in SB:
         j+=1
         typestr='_SB'
         lum_str='luminosity'+typestr
         mstar_str='stellar_mass'+typestr
         s1='SB '+s
         subplot_sim_LX(ax,df,s1,j,mstar_str,lum_str,markers_size,edge_width,markers_style,lambda_ave=lambda_ave,m_str='(9.5, 10.0]')
      if s in Q:
         j+=1
         typestr='_Q'
         lum_str='luminosity'+typestr
         mstar_str='stellar_mass'+typestr
         s2='Q '+s
         subplot_sim_LX(ax,df,s2,j,mstar_str,lum_str,markers_size,edge_width,markers_style,lambda_ave=lambda_ave,m_str='(10.0, 10.5]')
   
      if (not (Q or SB)) or (s in SF):
         if Q or SB or SF:
            s='SF '+s
         j+=1
         if (i==0) and ('(9.5, 10.0]' in df.index):
            m_str='(9.5, 10.0]'
         elif (i==1) and ('(9.0, 9.5]' in df.index):
            m_str='(9.0, 9.5]'
         elif (i==3) and ('(10.0, 10.5]' in df.index):
            m_str='(10.0, 10.5]'
         else:
            m_str=df.index[0]
         subplot_sim_LX(ax,df,s,j,markers_size=markers_size,edge_width=edge_width,markers_style=markers_style,lambda_ave=lambda_ave,m_str=m_str)

   #plt.text(0.83, 0.41, f'z = {z:.1f}', transform=fig.transFigure, **text_pars)
   if method_legend:
      ax.text(0.03, 0.965, method_legend, transform=ax.transAxes, **text_pars)
   if legend:
      leg=ax.legend(loc=legend_loc,title=leg_title,framealpha=0.4,ncol=ncol)
      leg._legend_box.align= "left"
   return

def subplot_sim_LX(ax,df,s,j=0,mstar_str='stellar_mass',lum_str='luminosity',markers_size=1,edge_width=1,markers_style=None,lambda_ave=False,m_str='(9.0, 9.5]'):
   # errorbars of bootstrapped simulation points
   yerr=np.array([df.loc[m_str:,(lum_str,0.5)] - df.loc[m_str:,(lum_str,0.05)], 
                  df.loc[m_str:,(lum_str,0.95)] - df.loc[m_str:,(lum_str,0.5)]])

   if markers_style:
      ax.scatter(df.loc[m_str:,(mstar_str,0.5)],df.loc[m_str:,(lum_str,0.5)], edgecolor='Black', label=s,color=cols[j+1],marker=markers_style[j], s=markers_size, linewidth=edge_width)
   else:
      ax.scatter(df.loc[m_str:,(mstar_str,0.5)],df.loc[m_str:,(lum_str,0.5)], edgecolors='Black', label=s,color=cols[j+1], s=markers_size, linewidth=edge_width)
   ax.errorbar(df.loc[m_str:,(mstar_str,0.5)],df.loc[m_str:,(lum_str,0.5)], 
                     yerr=yerr, linestyle=ls[j], zorder=0,color=cols[j+1])

   if lambda_ave:
      texts=[]
      for key, row in df.iterrows():
         lambda_ave_str='lambda_ave'
         if lum_str.rfind('_')>=0:
            lambda_ave_str+=lum_str[lum_str.rfind('_'):]
         if np.isnan(row.loc[(lambda_ave_str,'median')]):
            texts.append(ax.text(10, 1, ''))
         else:
            texts.append(ax.text(row.loc[(mstar_str,0.5)], row.loc[(lum_str,0.5)], f"{row.loc[(lambda_ave_str,'mean')]:.2f}", size=8))
      #adjust_text(texts)

def subplot_data_LX(ax,data, label, marker,m_min=2):
    ax.scatter(data['m_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i], edgecolors=cols[0], marker=marker,color='None',label=label,s=(plt.rcParams['lines.markersize']*1.4)**2)
    #ax.scatter(data['m_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i], edgecolors='Black', marker=marker,color=cols[0],label=label)
    ax.errorbar(data['m_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i],
                    yerr=np.array([data['l_ave'][0,m_min:,i] - data['l_ave'][2,m_min:,i], 
                        data['l_ave'][1,m_min:,i] - data['l_ave'][0,m_min:,i]]),
                    linestyle='solid',color=cols[0], zorder=0)
    return

#%%
# make 1 single comparison plot
def comp_plot(df_dic,method_legend=None,filename='Comparisons',leg_title=None,i=0,SB=[],Q=[],SF=[],m_min=2,legend_loc='lower right',markers_style=None,lambda_ave=False,ncol=1):
    fig,ax = plt.subplots(figsize=[9, 6])
    #plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")

    comp_subplot(ax,df_dic,method_legend,leg_title,i=i,Q=Q,SB=SB,SF=SF,m_min=m_min,legend_loc=legend_loc,markers_style=markers_style,lambda_ave=lambda_ave,ncol=ncol)
    
    ax.set_yscale('log')
    #ax.set_ylim(bottom=8e-4)
    #ax.set_xlim(9.,11.25)
    ax.set_xlabel('log$_{10}$<M$_*$> (M$_\odot$)')
    ax.set_ylabel('<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)')
    plt.savefig(curr_dir+'/Ros_plots/'+filename+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
    return
#%%
# Plot SFR vs LX
#def SFR_LX(df_dic,data,leg_title=None,m_min=2):
def SFR_LX(df_dic,data,leg_title=None,m_min=2,filename='SFvsLX',SB=[],Q=[],SF=[]):
   fig,ax = plt.subplots(figsize=[12, 8])
   
   SFR_LX_subplot(ax,df_dic,data,leg_title=leg_title,m_min=m_min,SB=SB,Q=Q,SF=SF)

   #plt.text(0.137, 0.78, f'z = {z:.1f}', transform=fig.transFigure, **text_pars)
   #plt.text(0.137, 0.865, f"z = {z:.1f}\nHalo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}", transform=fig.transFigure, **text_pars)
   ax.set_ylabel('<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)')
   plt.savefig(curr_dir+'/Ros_plots/'+filename+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 

def SFR_LX_subplot(ax,df_dic,data,leg_title=None,m_min=2,SB=[],Q=[],SF=[]):
   edge_width=0.5
   handles=[]

   # define parameters for datapoints colors and colorbar
   _min = np.nanmin(data['m_ave'][0,m_min:,:])
   _max = np.nanmax(data['m_ave'][0,m_min:,:])
   if Q:
      _min = np.minimum(np.nanmin(data_Q['m_ave'][0,4:,:]),_min)
      _max = np.maximum(np.nanmax(data_Q['m_ave'][0,4:,:]),_max)
   if SB:
      _min = np.minimum(np.nanmin(data_SB['m_ave'][0,:,:]),_min)
      _max = np.maximum(np.nanmax(data_SB['m_ave'][0,:,:]),_max)
   for s,bs_perc in df_dic.items():
      _min = np.minimum(np.nanmin(bs_perc['stellar_mass',0.5]),_min)
      _max = np.maximum(np.nanmax(bs_perc['stellar_mass',0.5]),_max)
   #plt.rcParams['figure.figsize'] = [12, 8]

   j=-1
   for s,bs_perc in df_dic.items():
      if s in SB:
         j+=1
         SFR_str='SFR_SB'
         lum_str='luminosity_SB'
         mstar_str='stellar_mass_Q'
         s1='SB '+s
         handles+=subplot_sim_SFR(ax,bs_perc,s1,_min,_max,j,SFR_str,lum_str,mstar_str,edge_width,m_str='(9.5, 10.0]')
      if s in Q:
         j+=1
         SFR_str='SFR_Q'
         lum_str='luminosity_Q'
         mstar_str='stellar_mass_Q'
         s2='Q '+s
         handles+=subplot_sim_SFR(ax,bs_perc,s2,_min,_max,j,SFR_str,lum_str,mstar_str,edge_width,m_str='(10.0, 10.5]')
      
      if (not (Q or SB)) or (s in SF):
         if Q or SB or SF:
            s='SF '+s
         j+=1
         SFR_str='SFR'
         lum_str='luminosity'
         mstar_str='stellar_mass'
         handles+=subplot_sim_SFR(ax,bs_perc,s,_min,_max,j,SFR_str,lum_str,mstar_str,edge_width)

   # "real" datapoints
   if Q or SB:
      label='SF Carraro et al. (2020)' 
   else: 
      label='Carraro et al. (2020)'
   handles+=subplot_data_SFR(ax,data, label,marker="s",_min=_min,_max=_max,m_min=m_min,edge_width=edge_width)

   if Q:
      label='Q Carraro et al. (2020)'
      handles+=subplot_data_SFR(ax,data_Q, label,marker="X",_min=_min,_max=_max,m_min=4,edge_width=edge_width)

   if SB:
      label='SB Carraro et al. (2020)'
      handles+=subplot_data_SFR(ax,data_SB, label,marker="d",_min=_min,_max=_max,m_min=0,edge_width=edge_width)

   sc=ax.scatter([],[], vmin = _min, vmax = _max)
   # colorbar, labels, legend, etc
   fig.colorbar(sc,pad=0.005).set_label('Stellar mass (M$_\odot$)')
   leg=ax.legend(handles=handles, loc='lower right',title=leg_title,handlelength=3)
   leg._legend_box.align= "left"
   #ax.legend(handles=handles, loc='lower right',title=leg_title,handlelength=3)
   ax.set_xlabel('<SFR> (M$_\odot$/yr)')
   ax.set_xscale('log')
   ax.set_yscale('log')
   return

def subplot_sim_SFR(ax,bs_perc,s,_min,_max,j=0,SFR_str='SFR',lum_str='luminosity',mstar_str='stellar_mass',edge_width=1,m_str='(9.0, 9.5]'):
   # errorbars of bootstrapped simulation points
   xerr=np.array([bs_perc.loc[m_str:,(SFR_str,0.5)] - bs_perc.loc[m_str:,(SFR_str,0.05)], 
               bs_perc.loc[m_str:,(SFR_str,0.95)] - bs_perc.loc[m_str:,(SFR_str,0.5)]])
   yerr=np.array([bs_perc.loc[m_str:,(lum_str,0.5)] - bs_perc.loc[m_str:,(lum_str,0.05)], 
                  bs_perc.loc[m_str:,(lum_str,0.95)] - bs_perc.loc[m_str:,(lum_str,0.5)]])

   # simulated datapoints
   data_pars=dict(marker=markers[j],linestyle=ls[j],color=cols[j+1], markeredgecolor='Black')
   ax.scatter(bs_perc.loc[m_str:,(SFR_str,0.5)],bs_perc.loc[m_str:,(lum_str,0.5)], vmin = _min, vmax = _max, marker=data_pars['marker'], edgecolors='Black',
               c=bs_perc.loc[m_str:,(mstar_str,0.5)] , s=bs_perc.loc[m_str:,(mstar_str,0.5)]*10,linewidth=edge_width)
   ax.errorbar(bs_perc.loc[m_str:,(SFR_str,0.5)],bs_perc.loc[m_str:,(lum_str,0.5)],
                  xerr=xerr, yerr=yerr, linestyle=data_pars['linestyle'], c=data_pars['color'], zorder=0)
   return [mlines.Line2D([], [], label=s, **data_pars)]

def subplot_data_SFR(ax,data, label, marker,_min,_max,m_min=2,edge_width=1):
   data_pars=dict(marker=marker,linestyle='-', markeredgecolor='Black', color=cols[0])
   ax.scatter(data['sfr_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i], vmin = _min, vmax = _max, edgecolors=data_pars['markeredgecolor'],
               c=data['m_ave'][0,m_min:,0], s=data['m_ave'][0,m_min:,0]*10, marker=data_pars['marker'],linewidth=edge_width)
   ax.errorbar(data['sfr_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i],
                  xerr=[data['sfr_ave'][0,m_min:,i]-data['sfr_ave'][2,m_min:,i],
                     data['sfr_ave'][1,m_min:,i]-data['sfr_ave'][0,m_min:,i]],
                  yerr=np.array([data['l_ave'][0,m_min:,i] - data['l_ave'][2,m_min:,i], 
                     data['l_ave'][1,m_min:,i] - data['l_ave'][0,m_min:,i]]),
                  linestyle=data_pars['linestyle'], zorder=0, color=data_pars['color'])
   handle = [mlines.Line2D([], [], label=label,**data_pars)]
   return handle

def generate_pie_markers(n=4):
   # generates n markers where each is a slice of pie chart
   # of the same size
   slice_size=1./n
   markers=[]
   for i in range(n):
      x = np.cos(2 * np.pi * np.linspace(i*slice_size, i*slice_size+slice_size))
      y = np.sin(2 * np.pi * np.linspace(i*slice_size, i*slice_size+slice_size))
      m = np.row_stack([[0, 0], np.column_stack([x, y]), [0, 0]])
      markers.append(m)

   return markers
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
params = {'legend.fontsize': 'medium',
          'legend.title_fontsize':'medium'}
plt.rcParams.update(params)


# make 4 comparison plots
fig,axs = plt.subplots(2,2,figsize=[15, 10], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

##########################################################
# top-left
paths = glob.glob(curr_dir+'/Ros_plots/Standard_R&V/bs_perc_z1.0_lambda-0.80_alpha0.10*.csv') + \
   sorted(glob.glob(curr_dir+f'/Ros_plots/Scaling_rels/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv') )
#print(paths)
keys=['Reines & Volonteri (2015)', 'Davis et al. (2018)', 'Sahu et al. (2019)', 'Shankar et al. (2016)']
#read DFs
df_dic = read_dfs(paths,keys)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}"
leg_title=r"M$_{\rm BH}$-M$_*$ scaling relation"
comp_subplot(axs[0,0],df_dic,leg_title=leg_title,i=i)
#comp_plot(df_dic,filename='Talk_scal_rels',leg_title=leg_title,i=i)

##########################################################
# top-right
# # Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+'/Ros_plots/Standard_R&V/bs_perc_z1.0_lambda-0.80_alpha0.10*.csv') + \
         sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}*.csv'))

#read DFs
df_dic = read_dfs(paths)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nDuty Cycle: {methods['duty_cycle']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"

#markers_style=[(q[0],   (plt.rcParams['lines.markersize']*1.2)**2,   1, 'left')]+\
#            2*[('o',   (plt.rcParams['lines.markersize']*1.2)**2,         1, 'full')]+\
#               [(q[1], (plt.rcParams['lines.markersize']*1.2)**2,    1, 'right')]+\
#               [(q[2],   (plt.rcParams['lines.markersize']*1.2)**2,      0.5, 'bottom')]+\
#               [(q[3],   (plt.rcParams['lines.markersize']*1.2)**2,      0.5, 'top')]+\
#            2*[('o',   (plt.rcParams['lines.markersize']*1.2)**2,         1, 'full')]
q = generate_pie_markers(4)
markers_style=[q[0]]+2*['o']+[q[1],q[2],q[3]]+1*['o']
leg_title='Eddington ratio distribution'
comp_subplot(axs[0,1],df_dic,leg_title=leg_title,i=i,lambda_ave=True,ncol=2)#,markers_style=markers_style
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}*.csv'))
df_dic = read_dfs(paths)
#comp_plot(df_dic,filename='Talk_Edd_distr',leg_title=leg_title,i=i,ncol=2)
##########################################################
# bottom-left
# Comparison of Davis slopes # but now it's R&V
# define DF path
paths = glob.glob(curr_dir+'/Ros_plots/Standard_R&V/bs_perc_z1.0_lambda-0.80_alpha0.10*.csv') + \
      sorted(glob.glob(curr_dir+f'/Ros_plots/Davis_slope/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv'))
#print(paths)

#keys=['Davis et al. (2018) extended',r'Slope $\beta=2.5$',r'Slope $\beta=2.0$',r'Slope $\beta=1.5$',r'Slope $\beta=1.0$']
keys=['Reines & Volonteri (2015)',r'Slope $\beta=1.5$',r'Slope $\beta=2.0$',r'Slope $\beta=2.5$',r'Slope $\beta=3.0$']
#read DFs
df_dic = read_dfs(paths,keys)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}"
leg_title=r"M$_{\rm BH}$-M$_*$ scaling relation"+"\nwith varying slope"
comp_subplot(axs[1,0],df_dic,leg_title=leg_title,i=i)
#comp_plot(df_dic,filename='Talk_Slopes',leg_title=leg_title,i=i)

##########################################################
# bottom-right
# Duty Cycle comparison
paths = glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv') + \
         glob.glob(curr_dir+f'/Ros_plots/Duty_Cycles/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv')
keys=['Schulze et al. (2015)','Georgakakis et al. (2017)','const=0.2','Man et al. (2019)']
#paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Test_dutycycle_onoff/bs_perc_z{z}_lambda0.10_alpha0.15*.csv'))
#keys=['Georgakakis et al. (2017)','Man et al. (2019)','Schulze et al. (2015)','const=0.2']

#print(paths)
#read DFs
df_dic = read_dfs(paths,keys)

#method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
markers_style=[q[3],q[2],q[1],q[0]]
leg_title=r'Duty cycle'
comp_subplot(axs[1,1],df_dic,leg_title=leg_title,i=i,markers_style=markers_style)
#comp_plot(df_dic,filename='Talk_duty_cycles',leg_title=leg_title,i=i)
##########################################################
# invisible labels for creating space:
#axs[-1, 0].set_xlabel('.', color=(0, 0, 0, 0))
#axs[-1, 0].set_ylabel('.', color=(0, 0, 0, 0))
# common lables:
fig.text(0.5, 0.07, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.08, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
#plt.ylim(2.1e-6,9e2)
plt.ylim(bottom=2e-4)
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
fig,axs = plt.subplots(1,3,figsize=[15, 5], sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})#, sharex=True
sigma=False # don't show the standard deviation of the gaussian in the legend of the plots
mean=True

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
#paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}*.csv') 
paths = sorted(paths)
method_legend=rf"M$_{{\rm BH}}$-M$_*$: {methods['BH_mass_method']}, slope $\beta=2$"
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma,mean=mean)

#leg_title=f"z = {z:.2f}"+"\n"+rf"M$_{{\rm BH}}$-M$_*$: {methods['BH_mass_method']}"
leg_title=f"z = {z:.2f}"
comp_subplot(axs[0],df_dic,method_legend=method_legend,m_min=3,leg_title=leg_title,i=i)#

##########################################################
# center
z = 1.
i=reds_dic.get(z)
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.5_sigma0.3*.csv*') + \
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.0_sigma0.3*.csv*')  + \
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.5_sigma0.3*.csv*')  
#paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma,mean=mean)

leg_title=f"z = {z:.2f}"
comp_subplot(axs[1],df_dic,leg_title=leg_title,m_min=2,i=i)#

##########################################################
# right
z = 2.7
i=reds_dic.get(z)
# # Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}*.csv') 
#paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma,mean=mean)

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
"""
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
        """
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_*.csv*')    + \
         glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter/bs_perc_z{z}_*.csv*') 
df_dic = read_dfs(paths)

method_legend=f'z = {z:.2f}\n'+r"M$_{\rm BH}$-M$_*$: "+f"{methods['BH_mass_method']}"
leg_title='Eddington ratio distribution'

comp_plot(df_dic,method_legend,filename='Comparison_lambdachar',leg_title=leg_title,i=i,legend_loc=(1.02,0),lambda_ave=True)

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
   paths =glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_*_sigma0.3*.csv*') 
   paths = sorted(paths)
   #read DFs
   df_dic = read_dfs(paths)
   cat_list=sorted(list(df_dic.keys()))

   comp_plot(df_dic,method_legend,filename=f'Comparison_SF',leg_title=leg_title,i=i,SF=cat_list,m_min=3,legend_loc=(1.02,0))
   comp_plot(df_dic,method_legend,filename=f'Comparison_Q',leg_title=leg_title,i=i,Q=cat_list,m_min=3,legend_loc=(1.02,0))
   comp_plot(df_dic,method_legend,filename=f'Comparison_SB',leg_title=leg_title,i=i,SB=cat_list,m_min=3,legend_loc=(1.02,0))

###################################################
######### Referee - Test lambda min ###############
###################################################
# plot global properties
plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")
params = {'legend.fontsize': 'small',
          'legend.title_fontsize':'small'}
plt.rcParams.update(params)

z=1.0
i=reds_dic.get(z)

q = generate_pie_markers(5)
markers_style=[q[0],q[1],q[2],q[3],q[4]]

# make 4 comparison plots
fig,axs = plt.subplots(2,2,figsize=[15, 10], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

##########################################################
# top-left
# Schulze 
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter/bs_perc_z{z}_alpha0.1_lambda-0.8*.csv') +\
      glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min/bs_perc_z{z}_lambda-0.80_alpha0.10*_Schulze*.csv'))
#print(paths)
#read DFs
df_dic = read_dfs(paths)
# remove common Eddington ratio distribution, which is the same for all:
dict_keys=list(df_dic.keys())
match = SequenceMatcher(None, dict_keys[0], dict_keys[1]).find_longest_match(0, len(dict_keys[0]), 0, len(dict_keys[1]))
#print(match)
match=dict_keys[1][match.a: match.a + match.size+2]
new_keys= [a_string.replace(match, "") for a_string in dict_keys]
new_keys[0]=r'$\log(\lambda_{min})=$-4'
df_dic = dict(zip(new_keys, list(df_dic.values())))

q = generate_pie_markers(5)
markers_style=[q[0],q[1],q[2],q[3],q[4]]


method_legend="Edd ratio distr: "+match[:-2]+'\n'+r"M$_{\rm BH}$-M$_*$: Reines & Volonteri (2015)"+f"\nz={z}"
leg_title="Schulze"
comp_subplot(axs[0,0],df_dic,leg_title=leg_title,i=i,method_legend=method_legend,markers_style=markers_style)

##########################################################
# top-right
# # Georgakakis17
paths = glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_lambda-0.80_alpha0.10*_Georgakakis*.csv') + \
         glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min/bs_perc_z{z}_lambda-0.80_alpha0.10*_Georgakakis*.csv')
paths = sorted(paths)

#read DFs
df_dic = read_dfs(paths)
dict_keys=list(df_dic.keys())
new_keys= [a_string.replace(match, "") for a_string in dict_keys]
new_keys[0]=r'$\log(\lambda_{min})=$ -4'
df_dic = dict(zip(new_keys, list(df_dic.values())))

leg_title='Georgakakis17'
comp_subplot(axs[0,1],df_dic,leg_title=leg_title,i=i,markers_style=markers_style)
##########################################################
# bottom-left
# Man19
paths = glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_lambda-0.80_alpha0.10*_Man*.csv') + \
      glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min/bs_perc_z{z}_lambda-0.80_alpha0.10*_Man*.csv')
paths = sorted(paths)

#read DFs
df_dic = read_dfs(paths)
dict_keys=list(df_dic.keys())
new_keys= [a_string.replace(match, "") for a_string in dict_keys]
new_keys[0]=r'$\log(\lambda_{min})=$ -4'
df_dic = dict(zip(new_keys, list(df_dic.values())))

leg_title=r"Man19"
comp_subplot(axs[1,0],df_dic,leg_title=leg_title,i=i,markers_style=markers_style)

##########################################################
# bottom-right
# constant=0.2
paths = glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_lambda-0.80_alpha0.10*_const0.2*.csv')+\
      glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min/bs_perc_z{z}_lambda-0.80_alpha0.10*_const0.2*.csv')
paths = sorted(paths)

#read DFs
df_dic = read_dfs(paths)
dict_keys=list(df_dic.keys())
new_keys= [a_string.replace(match, "") for a_string in dict_keys]
new_keys[0]=r'$\log(\lambda_{min})=$ -4'
df_dic = dict(zip(new_keys, list(df_dic.values())))

leg_title=r'$const=0.2$'
comp_subplot(axs[1,1],df_dic,leg_title=leg_title,i=i,markers_style=markers_style)
##########################################################
# invisible labels for creating space:
#axs[-1, 0].set_xlabel('.', color=(0, 0, 0, 0))
#axs[-1, 0].set_ylabel('.', color=(0, 0, 0, 0))
# common lables:
fig.text(0.5, 0.07, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.08, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
plt.ylim(2.1e-6,9e2)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/Test_lambda_min_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;


#%%
######################################################################
############ Compare SF,Q,SB LX-SFR relation ############
######################################################################
# Comparison of Eddington ratio distributions SFR vs LX for different galazy types
# define DF path
z = 1.0
i=reds_dic.get(z)
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_*.csv*') )
#read DFs
df_dic = read_dfs(paths)
cat_list=sorted(list(df_dic.keys()))
leg_title='Eddington ratio distribution'

SFR_LX(df_dic,data,leg_title, m_min = 4,filename='SFR_LX_SF',SF=cat_list)
SFR_LX(df_dic,data,leg_title, m_min = 4,filename='SFR_LX_Q',Q=cat_list)
SFR_LX(df_dic,data,leg_title, m_min = 4,filename='SFR_LX_SB',SB=cat_list)

#%%
######################################################################
############ Compare SF,Q,SB LX-SFR,LX-M* relations together ############
######################################################################
# make 1x2 comparison plots
mean=True
sigma=False
fig,axs = plt.subplots(1,2,figsize=[15, 5], sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
z = 1.0
i=reds_dic.get(z)

paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.5_sigma0.3*.csv*') +\
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.75_sigma0.3*.csv*') +\
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.0_sigma0.3*.csv*') +\
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.25_sigma0.3*.csv*') +\
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.5_sigma0.3*.csv*') +\
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-3.0_sigma0.3*.csv*') 
#paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}_mean-1.5_sigma0.3*.csv*') +\
#      glob.glob(curr_dir+f'/Ros_plots/Test_R&V_Gaussian_slope2_norm/bs_perc_z{z}_mean-1.50_sigma0.3*norm6.75.csv*') +\
#      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}_mean-1.75_sigma0.3*.csv*') +\
#      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}_mean-2.0_sigma0.3*.csv*') +\
#      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}_mean-2.25_sigma0.3*.csv*') +\
#      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}_mean-2.5_sigma0.3*.csv*') +\
#      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}_mean-3.0_sigma0.3*.csv*') 
#paths = sorted(paths)
print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma,mean=mean)
print(df_dic.keys())
sb_list=['Gaussian $\\mu=-1.50$; norm=6.75','Gaussian $\\mu=-2.00$']
q_list=['Gaussian $\\mu=-2.50$']
sf_list=['Gaussian $\mu=-1.50$']

method_legend=f"z={z}"+"\n"+r"M$_{\rm BH}$-M$_*$: Reines & Volonteri (2015)"
comp_subplot(axs[0],df_dic,method_legend,leg_title=leg_title,i=i,Q=q_list,SB=sb_list,SF=sf_list,m_min=2,legend=False,markers_style=markers)
axs[0].set_xlabel('<M$_*$> (M$_\odot$)')

######################################################################
# second subplot
#print(df_dic.keys())
leg_title='Eddington ratio distribution'

SFR_LX_subplot(axs[1],df_dic,data,leg_title, m_min = 2,SB=sb_list,Q=q_list,SF=sf_list)
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
paths =glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.50_sigma0.3*.csv*') +\
      glob.glob(curr_dir+f'/Ros_plots/Test_R&V_Gaussian__norm/bs_perc_z{z}_mean-1.50_sigma0.3*.csv*') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths)
SB_list=list(df_dic.keys())
#print(SB_list)

#print(SB_list[0], SB_list[1])
match = SequenceMatcher(None, SB_list[0], SB_list[1]).find_longest_match(0, len(SB_list[0]), 0, len(SB_list[1]))
#print(match)
match=SB_list[1][match.b: match.b + match.size]
#print(match)
SB_list=list(df_dic.keys())
SB_list= [a_string.replace(match, "") for a_string in SB_list]
SB_list[0]=r'Original, norm=7.45'
df_dic = dict(zip(SB_list, list(df_dic.values())))
method_legend=f"BH_mass: Reines & Volonteri (2015)\nEdd ratio distrib:"+match
leg_title=r'Normalization of M$_{\rm BH}$-M$_*$'

comp_plot(df_dic,method_legend,filename=f'Comparison_SB_norm',leg_title=leg_title,i=i,Q=[],SB=SB_list,m_min=3)#,legend_loc=(1.02,0)
#%%
##################################################################################
############ Compare SF,Q,SB LX-SFR relation varying R&V normalization ############
##################################################################################
# Comparison of Eddington ratio distributions SFR vs LX for different galaxy types
# define DF path
z = 1.0
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.50_sigma0.3*.csv.bak') +\
        glob.glob(curr_dir+f'/Ros_plots/Test_R&V_Gaussian_SFQSB/bs_perc_z{z}_mean-1.50_sigma0.30_*norm6.75.csv') +\
        glob.glob(curr_dir+f'/Ros_plots/Test_R&V_Gaussian_SFQSB/bs_perc_z{z}_mean-1.50_sigma0.30_*norm7.75.csv')
#read DFs
df_dic = read_dfs(paths,lambdac=True)

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
#paths = glob.glob(curr_dir+f'/Ros_plots/Standard/Scatter_Mh-M*_0.0/bs_perc_z{z}_lambda0.10_alpha0.15*.csv') +\
#        glob.glob(curr_dir+f'/Ros_plots/Standard/bs_perc_z{z}_lambda0.10_alpha0.15*.csv') +\
#        glob.glob(curr_dir+f'/Ros_plots/Standard/Scatter_Mh-M*_0.3/bs_perc_z{z}_lambda0.10_alpha0.15*.csv') +\
#        glob.glob(curr_dir+f'/Ros_plots/Standard/Scatter_Mh-M*_0.6/bs_perc_z{z}_lambda0.10_alpha0.15*.csv')
paths = glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/Scatter_Mh-M*_0.0/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv') +\
        glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv') +\
        glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/Scatter_Mh-M*_0.3/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv') +\
        glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/Scatter_Mh-M*_0.6/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv')
#read DFs
keys=['No scatter','Scatter 0.15 dex (default)','Scatter 0.30 dex','Scatter 0.60 dex']
df_dic = read_dfs(paths,keys)

#method_legend=f"z={z}\nBH_mass: Shankar+16"
method_legend=f"z={z}\nBH_mass: Reines&Volonteri15"
leg_title='Scatter in MH-M* relation'

q = generate_pie_markers(4)
markers_style=[q[0],q[1],q[2],q[3]]
comp_plot(df_dic,method_legend,filename='Comparison_Scatter_Mh-M*',leg_title=leg_title,i=i,markers_style=markers_style)

#%%
#####################################################
############ Test scatter Mbh-M* ############
#####################################################
# # Comparison between standard configuration and Shankar relation with intrinsic scatter
# or double or quadruple scatter
z = 1.0
i=reds_dic.get(z)
#paths = glob.glob(curr_dir+f'/Ros_plots/Standard/bs_perc_z{z}_lambda0.10_alpha0.15*.csv') +\
#        glob.glob(curr_dir+f'/Ros_plots/Standard/Scatter_Mbh-M*_2*/bs_perc_z{z}_lambda0.10_alpha0.15*.csv') +\
#        glob.glob(curr_dir+f'/Ros_plots/Standard/Scatter_Mbh-M*_4*/bs_perc_z{z}_lambda0.10_alpha0.15*.csv') 
paths = glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv') +\
        glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/Scatter_Mbh-M*_2*/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv') +\
        glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/Scatter_Mbh-M*_4*/bs_perc_z{z}_lambda-0.80_alpha0.10*.csv') 
#read DFs
keys=['0.5 dex (default)','1.0 dex','1.5 dex']
df_dic = read_dfs(paths,keys)

#method_legend=f"z={z}\nBH_mass: Shankar+16"
method_legend=f"z={z}\nBH_mass: Reines&Volonteri15"
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