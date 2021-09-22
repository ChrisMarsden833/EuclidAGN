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
all_data={}
for key, val in read_data.items():
    all_data[key]=np.copy(val)
    all_data[key][all_data[key] == 0.] = np.nan
   
read_data = readsav('./IDL_data/active_only/vars_EuclidAGN_90.sav')#,verbose=True
active_data={}
for key, val in read_data.items():
    active_data[key]=np.copy(val)
    active_data[key][active_data[key] == 0.] = np.nan
#print(data.keys())

# Q
read_data = readsav('./IDL_data/vars_EuclidAGN_90_Q.sav')#,verbose=True
all_data_Q={}
for key, val in read_data.items():
    all_data_Q[key]=np.copy(val)
    all_data_Q[key][all_data_Q[key] == 0.] = np.nan
   
read_data = readsav('./IDL_data/active_only/vars_EuclidAGN_90_Q.sav')#,verbose=True
active_data_Q={}
for key, val in read_data.items():
    active_data_Q[key]=np.copy(val)
    active_data_Q[key][active_data_Q[key] == 0.] = np.nan

# SB
read_data = readsav('./IDL_data/vars_EuclidAGN_90_SB.sav')#,verbose=True
all_data_SB={}
for key, val in read_data.items():
    all_data_SB[key]=np.copy(val)
    all_data_SB[key][all_data_SB[key] == 0.] = np.nan

read_data = readsav('./IDL_data/active_only/vars_EuclidAGN_90_SB.sav')#,verbose=True
active_data_SB={}
for key, val in read_data.items():
    active_data_SB[key]=np.copy(val)
    active_data_SB[key][active_data_SB[key] == 0.] = np.nan

# duty cycles from appendix Carraro+20
data_duty_cycles=np.array([[np.nan,np.nan,np.nan,19./(19+4244 ), 75./(2514 + 75), 117./(1214+117), 38./(239+ 38)],
                           [np.nan,np.nan,14./22159,56./(56+15431),175./(7957 +175), 370./(3888+370),142./(761+142)], 
                           [np.nan,np.nan,np.nan,47./(47+13451),177./(11334+177), 368./(5273+368),172./(954+172)],
                           [np.nan,np.nan,np.nan, 3./(3 +264  ), 98./(159  + 98), 166./(726 +166), 70./(66 + 70)] ])
data_duty_cycles_Q=np.array([[ 12./(12+867), 19./(19+830),14./(14+271)],
                              [ 19./(19+2594), 72./(72+3354),25./(25+1107)]]) 
data_duty_cycles_SB=np.array([[ 4./(58+4), 5./(6+5),3./(2+3)],
                              [ 7./(162+7), 10./(162+10),5./(43+5)]]) 

"""
m_sf=[10.21,10.70,11.14]
m_sb=[ 9.87,10.52,10.96]
m_q=[10.27,10.74,11.16]
plt.plot(m_sf,data_duty_cycles[0],label='starforming')
plt.plot(m_q,data_duty_cycles_Q[0],label='quiescent')
plt.plot(m_sb,data_duty_cycles_SB[0],label='starburst')
plt.legend()
plt.title('z=0.45')
plt.show()
#
plt.plot(m_sf,data_duty_cycles[1],label='starforming')
plt.plot(m_q,data_duty_cycles_Q[1],label='quiescent')
plt.plot(m_sb,data_duty_cycles_SB[1],label='starburst')
plt.legend()
plt.title('z=1.0')
plt.show()
"""
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
def read_dfs(paths,keys=None,sigma=True,mean=True,lambdac=False,duty=False):
    #read and place in dictionary
    if keys:
      dictionary={}
      for p,key in zip(paths,keys):
         df=pd.read_csv(p,header=[0,1],index_col=0)
         # percentile values in column names to float type instead of string
         #df.columns.set_levels(df.columns.levels[1].astype(float,errors='ignore'),level=1,inplace=True)
         df.columns=df.columns.set_levels(convert_index_to_float(df.columns.levels[1]),level=1)
         dictionary[key]=df
      return dictionary
    else:
      pattern=re.compile('([a-z]+)(\-*\d*\.\d*)') 
      df_dict={}
      for p in paths:
         df=pd.read_csv(p,header=[0,1],index_col=0)
         #df.columns.set_levels(df.columns.levels[1].astype(float,errors='ignore'),level=1,inplace=True)
         df.columns=df.columns.set_levels(convert_index_to_float(df.columns.levels[1]),level=1)
         pars=dict(re.findall(pattern, p))
         if '_SB.csv' in p:
            new_key='SB '
         elif '_Q.csv' in p:
            new_key='Q '
         else:
            new_key=''

         if 'mean' in p:
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
         
         if duty:
            if 'const' in pars.keys():
               new_key+=f"; const={float(pars['const']):.2f}"
            else:
               duties={'Geo':'Georgakakis et al. (2017)','Man':'Man et al. (2019)','Schulze':'Schulze et al. (2015)'}
               for d in duties.keys():
                  if d in p:
                     new_key+=f"; {duties[d]}"

         if (lambdac==True) and ('lambdac' in pars.keys()):
               new_key+=fr"; $\zeta_c={np.asarray(pars['lambdac'], dtype=np.float64):.2f}$"
               #new_key+=fr"; $\zeta_c={pars['lambdac']:.2f}$"
         if 'lambdam' in pars.keys():
               new_key+=r"; $\log(\lambda_{min})=$"
               new_key+=f"{np.asarray(pars['lambdam'], dtype=np.float64):.0f}"
         if 'scatter' in pars.keys():
               new_key+=f"; scatter={float(pars['scatter']):.1f}"

         df_dict[new_key]=df
      return df_dict

#%%
# function for comparison subplots
def comp_subplot(ax,df_dic,method_legend=None,leg_title=None,m_min=3,i=0,
                  SB=[],Q=[],SF=[],legend=True,legend_loc='lower right',markers_style=None,lambda_ave=False,ncol=1,active=False):
   if active:
      data=active_data
      data_Q=active_data_Q
      data_SB=active_data_SB
   else:
      data=all_data
      data_Q=all_data_Q
      data_SB=all_data_SB
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

def subplot_sim_LX(ax,df,s,j=0,mstar_str='stellar_mass',lum_str='luminosity',markers_size=1,edge_width=1,markers_style=None,lambda_ave=False,m_str='(9.5, 10.0]'):
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

def subplot_data_LX(ax,data, label, marker,m_min=3):
    ax.scatter(data['m_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i], edgecolors=cols[0], marker=marker,color='None',label=label,s=(plt.rcParams['lines.markersize']*1.4)**2)
    #ax.scatter(data['m_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i], edgecolors='Black', marker=marker,color=cols[0],label=label)
    ax.errorbar(data['m_ave'][0,m_min:,i], data['l_ave'][0,m_min:,i],
                    yerr=np.array([data['l_ave'][0,m_min:,i] - data['l_ave'][2,m_min:,i], 
                        data['l_ave'][1,m_min:,i] - data['l_ave'][0,m_min:,i]]),
                    linestyle='solid',color=cols[0], zorder=0)
    return

#%%
# make 1 single comparison plot
def comp_plot(df_dic,method_legend=None,filename='Comparisons',leg_title=None,i=0,SB=[],Q=[],SF=[],m_min=3,legend_loc='lower right',markers_style=None,lambda_ave=False,ncol=1,active=False):
    fig,ax = plt.subplots(figsize=[9, 6])
    #plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")

    comp_subplot(ax,df_dic,method_legend,leg_title,i=i,Q=Q,SB=SB,SF=SF,m_min=m_min,legend_loc=legend_loc,markers_style=markers_style,lambda_ave=lambda_ave,ncol=ncol,active=active)
    
    ax.set_yscale('log')
    #ax.set_ylim(bottom=8e-4)
    #ax.set_xlim(9.,11.25)
    ax.set_xlabel('log$_{10}$<M$_*$> (M$_\odot$)')
    ax.set_ylabel('<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)')
    plt.savefig(curr_dir+'/Ros_plots/'+filename+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
    plt.close(fig)
    return
#%%
# Plot SFR vs LX
#def SFR_LX(df_dic,data,leg_title=None,m_min=2):
def SFR_LX(df_dic,leg_title=None,m_min=3,filename='SFvsLX',SB=[],Q=[],SF=[],active=False):
   fig,ax = plt.subplots(figsize=[12, 8])
   
   SFR_LX_subplot(ax,df_dic,leg_title=leg_title,m_min=m_min,SB=SB,Q=Q,SF=SF,active=active)

   #plt.text(0.137, 0.78, f'z = {z:.1f}', transform=fig.transFigure, **text_pars)
   #plt.text(0.137, 0.865, f"z = {z:.1f}\nHalo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}", transform=fig.transFigure, **text_pars)
   ax.set_ylabel('<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)')
   plt.savefig(curr_dir+'/Ros_plots/'+filename+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 
   plt.close(fig)

def SFR_LX_subplot(ax,df_dic,leg_title=None,m_min=3,SB=[],Q=[],SF=[],active=False):
   if active:
      data=active_data
      data_Q=active_data_Q
      data_SB=active_data_SB
   else:
      data=all_data
      data_Q=all_data_Q
      data_SB=all_data_SB

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
   sc.set_clim(_min,_max)
   print('mmin', _min, 'mmax',_max)
   # colorbar, labels, legend, etc
   fig.colorbar(sc,ax=ax,pad=0.005).set_label('Stellar mass (M$_\odot$)')
   leg=ax.legend(handles=handles, loc='lower right',title=leg_title,handlelength=3)
   leg._legend_box.align= "left"
   #ax.legend(handles=handles, loc='lower right',title=leg_title,handlelength=3)
   ax.set_xlabel('<SFR> (M$_\odot$/yr)')
   ax.set_xscale('log')
   ax.set_yscale('log')
   return

def subplot_sim_SFR(ax,bs_perc,s,_min,_max,j=0,SFR_str='SFR',lum_str='luminosity',mstar_str='stellar_mass',edge_width=1,m_str='(9.5, 10.0]'):
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

def subplot_data_SFR(ax,data, label, marker,_min,_max,m_min=3,edge_width=1):
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

def plot_dutycycle(df_dic,filename='measured_duty_cycles',i=0,m_str='(9.5, 10.0]',mstar_str='stellar_mass',m_min=3,legend_loc='lower right',method_legend=None,leg_title=None,markers_style=None):
   fig,ax = plt.subplots(figsize=[9, 6])
    #plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")

   subplot_dutycycle(df_dic,ax=ax,i=i,m_str=m_str,mstar_str=mstar_str,m_min=m_min,legend_loc=legend_loc,method_legend=method_legend)

   ax.set_yscale('log')
   ax.set_xlabel('log$_{10}$<M$_*$> (M$_\odot$)')
   ax.set_ylabel('AGN fraction')
   plt.savefig(curr_dir+'/Ros_plots/'+filename+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 
   plt.close(fig);
   return

def subplot_dutycycle(df_dic,ax,i=0,m_str='(9.5, 10.0]',mstar_str='stellar_mass',m_min=3,legend_loc='lower right',method_legend=None,markers_style=None,leg_title=None):
   markers_size=(plt.rcParams['lines.markersize']*1.2)**2
   edge_width=0.5

   # data U
   ax.scatter(active_data['m_ave'][0,m_min:,i], data_duty_cycles[i][m_min:], edgecolors=cols[0], marker='s',label='Carraro et al. (2020)',color='None')#

   # mocks U
   for j,(s,df) in enumerate(df_dic.items()):
      if markers_style:
         df.loc[m_str:].plot.scatter(x=(mstar_str,0.5),y=('AGN_fraction','Unnamed: 16_level_1'),edgecolors='Black', label=s,color=cols[j], s=markers_size, linewidth=edge_width,ax=ax,marker=markers_style[j])
      else:
         df.loc[m_str:].plot.scatter(x=(mstar_str,0.5),y=('AGN_fraction','Unnamed: 16_level_1'),edgecolors='Black', label=s,color=cols[j], s=markers_size, linewidth=edge_width,ax=ax)
      df.loc[m_str:].plot(x=(mstar_str,0.5),y=('AGN_fraction','Unnamed: 16_level_1'), color=cols[j],ax=ax,linestyle=ls[j],alpha=0.5,label='')

   if method_legend:
      ax.text(0.03, 0.965, method_legend, transform=ax.transAxes, **text_pars)

   leg=ax.legend(loc=legend_loc,framealpha=0.4,title=leg_title)#,title=leg_title,ncol=ncol
   leg._legend_box.align= "left"

   ax.set_yscale('log')
   return

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

###############################
######### Fig Test ###############
###############################
# figure to test results vs Viola    
z = 1.
i=reds_dic.get(z) # needed for IDL data
paths = glob.glob(curr_dir+'/Ros_plots/05_Wrong_scatter/Standard_R&V/Simple_average/bs_perc_z1.0_lambda-2.35_alpha0.10_lambdac-2.381.csv') + \
      glob.glob(curr_dir+'/Ros_plots/05_Wrong_scatter/Standard_R&V/Weighted_bootstrap/bs_perc_z1.0_lambda-2.35_alpha0.10_lambdac-2.381.csv') + \
      glob.glob(curr_dir+'/Ros_plots/05_Wrong_scatter/Standard_R&V/Subtract_XRBlum/bs_perc_z1.0_lambda-2.35_alpha0.10_lambdac-2.381.csv') + \
      glob.glob(curr_dir+f'/Ros_plots/05_Wrong_scatter/R&V_Gaussian/bs_perc_z1.0_mean-2.5_sigma0.3_lambdac-2.40.csv') #+\
      #glob.glob(curr_dir+'/Ros_plots/Standard_R&V/bs_perc_z1.0_lambda-2.35_alpha0.10_lambdac-2.381_noduty.csv') + \
      #glob.glob(curr_dir+'/Ros_plots/R&V_Schechter/bs_perc_z1.0_alpha0.1_lambda-2.35_lambdac-2.615.csv') 
#print(paths)
#read DFs
keys=['Schechter $x*=-2.35$, $\\alpha=0.10$','Schechter $x*=-2.35$, $\\alpha=0.10$ bootstrap','Schechter $x*=-2.35$, $\\alpha=0.10$ Lx - Lxrb','Gaussian $\\mu=-2.50$, $\\sigma=0.30$']
df_dic = read_dfs(paths,keys)
#print(df_dic.keys())

comp_plot(df_dic,filename='Compare_w_Viola',i=i)
#%%
###################################
###### Measured Duty cycles #######
###################################
########
# duty cycle plots of all tested Edd ratios 
# don't need to run these everytime
"""
zees = [1.,0.45]
for z in zees:
   i=reds_dic.get(z) # needed for IDL data
   paths = sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_*active*.csv'))

   for path in paths:
      df_dic = read_dfs([path],lambdac=True)
      plot_dutycycle(df_dic,filename=f'R&V_Gaussian/duty_cycle_'+path[path.rfind('/')+1:path.rfind('.csv')-1],i=i)


   paths = sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter/bs_perc_z{z}_*active*.csv'))

   for path in paths:
      df_dic = read_dfs([path],lambdac=True)
      plot_dutycycle(df_dic,filename=f'R&V_Schechter/duty_cycle_'+path[path.rfind('/')+1:path.rfind('.csv')-1],i=i)
"""
#######
z = 1.
i=reds_dic.get(z) # needed for IDL data
# make 4 comparison plots
fig,axs = plt.subplots(3,1,figsize=[5, 10], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

folders=['Duty_cycles','Duty_cycles_NH23','Duty_cycles_noNH']
strings=['Cut at $\log(N_H)=24$','Cut at $\log(N_H)=23$','No cut in $N_H$']
for j,(f,s) in enumerate(zip(folders,strings)):
   paths = sorted(glob.glob(curr_dir+f'/Ros_plots/{f}/bs_perc_z{z}_*_active*.csv'))
   #print(paths)
   keys=['Georgakakis et al. (2017)','Man et al. (2019)','Schulze et al. (2015)','const=0.2']

   df_dic = read_dfs(paths,keys,lambdac=True)
   subplot_dutycycle(df_dic,axs[j],i=i,legend_loc=(1.02,0),method_legend=s)

#fig.text(0.5, 0.07, r'$\log_{10}$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
#fig.text(0.08, 0.5, 'output U', va='center', ha='center', rotation='vertical',size='x-large')
plt.savefig(curr_dir+'/Ros_plots/measured_duty_cycles_vsNH'+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 
plt.close(fig);

#######
#######
z = 1.
i=reds_dic.get(z) # needed for IDL data
best_duties_10=['alpha-1.5_lambda-2.0',
'alpha-1.5_lambda-1.8',
'alpha-1.0_lambda-1.8',
'alpha-0.5_lambda-1.6',
'alpha-0.4_lambda-1.6',
'alpha-0.3_lambda-1.6',
'alpha-0.2_lambda-1.6',
'mean-1.5_sigma0.3',
'mean-1.75_sigma0.3',
'mean-2.0_sigma0.5',
'mean-2.0_sigma0.7']

best_duties_45=['alpha-0.2_lambda-2.5',
'alpha-0.3_lambda-2.5',
'alpha-0.4_lambda-2.5',
'alpha-0.5_lambda-2.5',
'alpha0.2_lambda-2.5',
'alpha0.3_lambda-2.5',
'alpha0.4_lambda-2.5',
'mean-2.5_sigma0.3']

duties_list=[best_duties_45,best_duties_10]

########### 
# U Schulze
zees=[0.45,1.0]
for z,best_duties in zip(zees,duties_list):
   i=reds_dic.get(z) # needed for IDL data
   paths=[]
   paths_active=[]
   for d in best_duties:
      paths+=glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_{d}*_all.csv')
      paths+=glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter/bs_perc_z{z}_{d}*_all.csv')
      paths_active+=glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_{d}*_active.csv')
      paths_active+=glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter/bs_perc_z{z}_{d}*_active.csv')
   paths=sorted(paths)
   df_dic = read_dfs(paths,lambdac=True)
   comp_plot(df_dic,i=i,filename=f'Best_mdc_LX_M',lambda_ave=True,legend_loc=(1.02,0))

   paths_active=sorted(paths_active)
   df_dic = read_dfs(paths_active,lambdac=True)
   plot_dutycycle(df_dic,i=i,filename=f'Best_measured_duty_cycles',legend_loc=(1.02,0))
   comp_plot(df_dic,i=i,filename=f'Best_mdc_LX_M_active',lambda_ave=True,active=True,legend_loc=(1.02,0))

########### 
# U const=0.2
for z in zees:
   i=reds_dic.get(z) # needed for IDL data
   paths=[]
   paths_active=[]
   for d in best_duties:
      paths+=glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_Uconst02/bs_perc_z{z}_{d}*_all.csv')
      paths+=glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter_Uconst02/bs_perc_z{z}_{d}*_all.csv')
      paths_active+=glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_Uconst02/bs_perc_z{z}_{d}*_active.csv')
      paths_active+=glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter_Uconst02/bs_perc_z{z}_{d}*_active.csv')
   paths=sorted(paths)
   df_dic = read_dfs(paths,lambdac=True)
   comp_plot(df_dic,i=i,filename=f'Best_mdc_LX_M_Uconst02',lambda_ave=True,legend_loc=(1.02,0))

   paths_active=sorted(paths_active)
   df_dic = read_dfs(paths_active,lambdac=True)
   plot_dutycycle(df_dic,i=i,filename=f'Best_measured_duty_cycles_Uconst02',legend_loc=(1.02,0))
   comp_plot(df_dic,i=i,filename=f'Best_mdc_LX_M_active_Uconst02',lambda_ave=True,active=True,legend_loc=(1.02,0))

#%%
####################################
### Duty cycle effects on active ###
####################################
z = 0.45
i=reds_dic.get(z) # needed for IDL data
# make 4 comparison plots
fig,axs = plt.subplots(3,2,figsize=[15, 15], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

##########################################################
# top-left   
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_mean-2.00_sigma0.3*_active*const*.csv'))
#print(paths)
#read DFs
df_dic = read_dfs(paths,duty=True,mean=False,sigma=False)

method_legend=f"Active (logLX>42), simple mean, no NH cut"
q = generate_pie_markers(5)
markers_style=[q[1],q[2],q[0],q[3],q[4]]*4
leg_title=r'Duty cycle'
comp_subplot(axs[0,0],df_dic,leg_title=leg_title,i=i,active=True,markers_style=markers_style,m_min=3,legend_loc='upper left')#,method_legend=method_legend
##########################################################
# top-right   
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles_testscatter/bs_perc_z{z}_mean-2.00_sigma0.3*_active*const*scatter0.8.csv'))
#print(paths)
#read DFs
df_dic = read_dfs(paths,duty=True,mean=False,sigma=False)

method_legend=f"Active (logLX>42), simple mean, no NH cut"
leg_title=r'Duty cycle'
comp_subplot(axs[0,1],df_dic,leg_title=leg_title,i=i,active=True,markers_style=markers_style,m_min=3,legend_loc='upper left')#,method_legend=method_legend
##########################################################
# mid-left   
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles_testscatter/bs_perc_z{z}_mean-2.00_sigma0.3*_active*const*scatter1.0.csv'))
#print(paths)
#read DFs
df_dic = read_dfs(paths,duty=True,mean=False,sigma=False)

method_legend=f"Active (logLX>42), simple mean, no NH cut"
leg_title=r'Duty cycle'
comp_subplot(axs[1,0],df_dic,leg_title=leg_title,i=i,active=True,markers_style=markers_style,m_min=3,legend_loc='upper left')#,method_legend=method_legend
##########################################################
# mid-right
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_mean-2.00_sigma0.3*_active*Man*.csv'))
paths += sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles_testscatter/bs_perc_z{z}_mean-2.00_sigma0.3*_active*Man19*.csv'))

#print(paths)
#read DFs
df_dic = read_dfs(paths,duty=True,mean=False,sigma=False)

method_legend=f"Active (logLX>42), simple mean, NH cut"
leg_title=r'Duty cycle'
comp_subplot(axs[1,1],df_dic,leg_title=leg_title,i=i,active=True,m_min=3,legend_loc='upper left')#,markers_style=markers_style)#,method_legend=method_legend
##########################################################
# bottom-left
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_mean-2.00_sigma0.3*_active*Geo*.csv'))
paths += sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles_testscatter/bs_perc_z{z}_mean-2.00_sigma0.3*_active*Geo*.csv'))
#print(paths)
#read DFs
df_dic = read_dfs(paths,duty=True,mean=False,sigma=False)

#method_legend=f"Active (logLX>42), weighted mean, no NH cut"
leg_title=r"M$_{\rm BH}$-M$_*$ scaling relation"
comp_subplot(axs[2,0],df_dic,leg_title=leg_title,i=i,active=True,m_min=3,legend_loc='upper left')#,method_legend=method_legend

##########################################################
# bottom-right
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_mean-2.00_sigma0.3*_active*Schulze*.csv'))
paths += sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles_testscatter/bs_perc_z{z}_mean-2.00_sigma0.3*_active*Schulze*.csv'))
#print(paths)

#read DFs
df_dic = read_dfs(paths,duty=True,mean=False,sigma=False)

#method_legend=f"Active (logLX>42), weighted mean, NH cut"
comp_subplot(axs[2,1],df_dic,leg_title=leg_title,i=i,active=True,m_min=3,legend_loc='upper left')#,method_legend=method_legend

##########################################################
# common lables:
fig.text(0.5, 0.07, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.08, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
#plt.ylim(2e0,8e1)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/scatter_effects_active_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 
plt.close(fig);

#%%
#################################
### Duty cycle effects on all ###
#################################
z = 1.
i=reds_dic.get(z) # needed for IDL data
# make 4 comparison plots
fig,axs = plt.subplots(2,2,figsize=[15, 10], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

##########################################################
# top-left   
# simple mean, all no NH cut
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles_testmean_noNH/bs_perc_z{z}_mean-2.25_sigma0.3*_all*.csv'))
#print(paths)
#read DFs
keys=['Georgakakis et al. (2017)','Man et al. (2019)','Schulze et al. (2015)','const=0.2']
df_dic = read_dfs(paths,keys,lambdac=True)

method_legend=f"All galaxies, simple mean, no NH cut"
q = generate_pie_markers(4)
markers_style=[q[3],q[2],q[1],q[0]]
leg_title=r'Duty cycle'
comp_subplot(axs[0,0],df_dic,leg_title=leg_title,i=i,method_legend=method_legend,active=False,markers_style=markers_style)
##########################################################
# top-right
# simple mean, all no NH cut
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles_testmean/bs_perc_z{z}_mean-2.25_sigma0.3*_all*.csv'))

#print(paths)
#read DFs
df_dic = read_dfs(paths,keys)

method_legend=f"All galaxies, simple mean, NH cut"
leg_title=r'Duty cycle'
comp_subplot(axs[0,1],df_dic,leg_title=leg_title,i=i,legend=False,active=False,method_legend=method_legend,markers_style=markers_style)
##########################################################
# bottom-left
paths =sorted( glob.glob(curr_dir+f'/Ros_plots/Duty_cycles_noNH/bs_perc_z{z}_*_all*.csv'))
print(keys)
print(paths)
#read DFs
df_dic = read_dfs(paths,keys)

method_legend=f"All galaxies, mean with duty cycle, no NH cut"
leg_title=r"M$_{\rm BH}$-M$_*$ scaling relation"
comp_subplot(axs[1,0],df_dic,leg_title=leg_title,i=i,legend=False,active=False,method_legend=method_legend)

##########################################################
# bottom-right
# Comparison of Davis slopes # but now it's R&V
paths = sorted(glob.glob(curr_dir+'/Ros_plots/Duty_cycles/bs_perc_z1.0_*_all*.csv') )
print(paths)

#read DFs
df_dic = read_dfs(paths,keys)

method_legend=f"All galaxies, mean with duty cycle, NH cut"
comp_subplot(axs[1,1],df_dic,leg_title=leg_title,i=i,legend=False,active=False,method_legend=method_legend)

##########################################################
# common lables:
fig.text(0.5, 0.07, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.08, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
#plt.ylim(2e0,8e1)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/duty_cycle_effects_all_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 
plt.close(fig);

#%%
"""
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

reds=[0.45,1.0]
for z in reds:
   i=reds_dic.get(z) # needed for IDL data
   # make 4 comparison plots
   fig,axs = plt.subplots(2,2,figsize=[15, 10], sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

   ##########################################################
   # top-left   
   # # Gaussian width Edd ratio distributions comparison
   if z==0.45: 
      standard_str="_mean-2.0_sigma0.3"
      schechter_str="_alpha-0.3_lambda-2.0"
   else:
      standard_str="_mean-1.5_sigma0.3"
      schechter_str="_alpha-0.3_lambda-2.25"
   paths = sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.0_sigma0.3*_all.csv'))+ \
            glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter/bs_perc_z{z}{schechter_str}*_all.csv') + \
            sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}{standard_str}*_all.csv'))+ \
            sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.5*_all.csv'))
            #glob.glob(curr_dir+'/Ros_plots/R&V_Schechter/bs_perc_z1.0_alpha1.5_lambda0.0*.csv') + \

   #print(paths)
   #read DFs
   df_dic = read_dfs(paths,lambdac=True)

   #method_legend=f"Halo to M*: {methods['halo_to_stars']}\nDuty Cycle: {methods['duty_cycle']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"

   #markers_style=[(q[0],   (plt.rcParams['lines.markersize']*1.2)**2,   1, 'left')]+\
   #            2*[('o',   (plt.rcParams['lines.markersize']*1.2)**2,         1, 'full')]+\
   #               [(q[1], (plt.rcParams['lines.markersize']*1.2)**2,    1, 'right')]+\
   #               [(q[2],   (plt.rcParams['lines.markersize']*1.2)**2,      0.5, 'bottom')]+\
   #               [(q[3],   (plt.rcParams['lines.markersize']*1.2)**2,      0.5, 'top')]+\
   #            2*[('o',   (plt.rcParams['lines.markersize']*1.2)**2,         1, 'full')]
   q = generate_pie_markers(2)
   markers_style=1*['o']+[q[0]]+[q[1]]+1*['o']
   leg_title='Eddington ratio distribution'
   comp_subplot(axs[0,0],df_dic,leg_title=leg_title,i=i,markers_style=markers_style)#,lambda_ave=True,ncol=2
   paths = sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}*.csv'))
   df_dic = read_dfs(paths)
   #comp_plot(df_dic,filename='Talk_Edd_distr',leg_title=leg_title,i=i,ncol=2)

   ##########################################################
   # top-right
   # Duty Cycle comparison
   #paths = glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/bs_perc_z{z}_mean-2.25_sigma0.3*.csv') + \
   #         glob.glob(curr_dir+f'/Ros_plots/Duty_Cycles/bs_perc_z{z}_mean-2.25_sigma*.csv')
   #keys=['Schulze et al. (2015)','Georgakakis et al. (2017)','const=0.2','Man et al. (2019)']
   paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_Cycles/bs_perc_z{z}_mean-2.25_sigma*_all*.csv'))
   keys=['Georgakakis et al. (2017)','Man et al. (2019)','Schulze et al. (2015)','const=0.2']

   print(paths)
   #read DFs
   df_dic = read_dfs(paths,keys)

   #method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nBH_mass: {methods['BH_mass_method']}\nBol corr: {methods['bol_corr']}"
   q = generate_pie_markers(4)
   markers_style=[q[3],q[2],q[1],q[0]]
   leg_title=r'Duty cycle'
   comp_subplot(axs[0,1],df_dic,leg_title=leg_title,i=i,markers_style=markers_style)
   #comp_plot(df_dic,filename='Talk_duty_cycles',leg_title=leg_title,i=i)
   ##########################################################
   # bottom-left
   paths = glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/bs_perc_z{z}_mean-2.25_sigma0.3*all*.csv') + \
         sorted(glob.glob(curr_dir+f'/Ros_plots/Scaling_rels/bs_perc_z{z}_mean-2.25_sigma0.3*all*.csv'))
   #print(paths)
   keys=['Reines & Volonteri (2015)', 'Davis et al. (2018)', 'Sahu et al. (2019)', 'Shankar et al. (2016)','R&V test']
   #read DFs
   df_dic = read_dfs(paths,keys)

   #method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}"
   leg_title=r"M$_{\rm BH}$-M$_*$ scaling relation"
   comp_subplot(axs[1,0],df_dic,leg_title=leg_title,i=i)
   #comp_plot(df_dic,filename='Talk_scal_rels',leg_title=leg_title,i=i)

   ##########################################################
   # bottom-right
   # Comparison of Davis slopes # but now it's R&V
   # define DF path
   paths = glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/bs_perc_z{z}_mean-2.25_sigma0.3*all*.csv') + \
         sorted(glob.glob(curr_dir+f'/Ros_plots/Davis_slope/bs_perc_z{z}_mean-2.25_sigma0.3*all*_slope2.0.csv'))+ \
         sorted(glob.glob(curr_dir+f'/Ros_plots/Davis_slope/bs_perc_z{z}_mean-2.25_sigma0.3*all*_slope3.0.csv'))
   #print(paths)

   #keys=['Davis et al. (2018) extended',r'Slope $\beta=2.5$',r'Slope $\beta=2.0$',r'Slope $\beta=1.5$',r'Slope $\beta=1.0$']
   keys=['Reines & Volonteri (2015)',r'Slope $\beta=2.0$',r'Slope $\beta=3.0$']
   #read DFs
   df_dic = read_dfs(paths,keys)

   #method_legend=f"Halo to M*: {methods['halo_to_stars']}\nEddington ratio: {methods['edd_ratio']}\nDuty Cycle: {methods['duty_cycle']}\nBol corr: {methods['bol_corr']}"
   leg_title=r"M$_{\rm BH}$-M$_*$ scaling relation"+"\nwith varying slope"
   comp_subplot(axs[1,1],df_dic,leg_title=leg_title,i=i)
   #comp_plot(df_dic,filename='Talk_Slopes',leg_title=leg_title,i=i)

   ##########################################################
   # invisible labels for creating space:
   #axs[-1, 0].set_xlabel('.', color=(0, 0, 0, 0))
   #axs[-1, 0].set_ylabel('.', color=(0, 0, 0, 0))
   # common lables:
   fig.text(0.5, 0.07, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
   fig.text(0.08, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
   #plt.ylim(2e0,8e1)
   plt.yscale('log')
   plt.savefig(curr_dir+f'/Ros_plots/fig2_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 
   plt.close(fig);

z=1.0
i=reds_dic.get(z) # needed for IDL data
#%%
####################################################
##### Test of the mean on AGN-SF subsamples ######
####################################################
z = 0.45
i=reds_dic.get(z)
fig,axs = plt.subplots(1,2,figsize=[15, 5], sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})#, sharex=True
sigma=False # don't show the standard deviation of the gaussian in the legend of the plots
mean=False
##########################################################
# left plot:
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_mean-2.00_sigma0.3*_active*.csv'))
#print('###########################')
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma,mean=mean,duty=True)

leg_title=f"Duty cycles with Eq.2 (weighted luminosity)"
comp_subplot(axs[0],df_dic,leg_title=leg_title,m_min=3,i=i,active=True,legend_loc='upper left')#

##########################################################
# right plot:
# find a zeta_c that represents each mass
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Duty_cycles_testsubsamples/bs_perc_z{z}_mean-2.00_sigma0.3*_active*.csv'))
df_dic = read_dfs(paths,sigma=sigma,mean=mean,duty=True)

leg_title=f"Duty cycles with Eq.3 (mean luminosity) on random subsamples"
comp_subplot(axs[1],df_dic,leg_title=leg_title,m_min=3,i=i,active=True,legend_loc='upper left')#
##########################################################
# common lables:
fig.text(0.5, 0.04, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.07, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
#plt.ylim(5e-4,1.8e2)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/test_mean_subsamples.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
plt.close(fig)
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
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.5_sigma0.3*all.csv') +\
   glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-3.0_sigma0.3*all.csv')
paths = sorted(paths)
method_legend=rf"M$_{{\rm BH}}$-M$_*$: {methods['BH_mass_method']}"
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma,mean=mean,lambdac=True)

#leg_title=f"z = {z:.2f}"+"\n"+rf"M$_{{\rm BH}}$-M$_*$: {methods['BH_mass_method']}"
leg_title=f"z = {z:.2f}"
comp_subplot(axs[0],df_dic,method_legend=method_legend,m_min=3,leg_title=leg_title,i=i)#

##########################################################
# center
z = 1.
i=reds_dic.get(z)
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.0_sigma0.3*all.csv*')  + \
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.25_sigma0.3*all.csv*')#+ \
      #glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.25_sigma0.3*.csv*')   
#paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma,mean=mean,lambdac=True)

leg_title=f"z = {z:.2f}"
comp_subplot(axs[1],df_dic,leg_title=leg_title,m_min=2,i=i)#

##########################################################
# right
z = 2.7
i=reds_dic.get(z)
# # Gaussian width Edd ratio distributions comparison
paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.25_sigma0.3*all.csv*')  + \
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-1.5_sigma0.3*all.csv*')
#paths = glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian_slope2/bs_perc_z{z}*.csv') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma,mean=mean,lambdac=True)

leg_title=f"z = {z:.2f}"
comp_subplot(axs[2],df_dic,leg_title=leg_title,m_min=4,i=i)
##########################################################
# common lables:
fig.text(0.5, 0.04, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.07, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
plt.ylim(5e-4,1.8e2)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/fig3.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
plt.close(fig)

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

######### Gaussian
method_legend=f'z = {z:.2f}\n'+r"M$_{\rm BH}$-M$_*$: Reines & Volonteri (2015)"
leg_title='Eddington ratio distribution'
zees=[0.45,1.0,2.7]
# # Comparison between LX-M* relation of SF,Q,SB galaxies
for z in zees:
   i=reds_dic.get(z)
   paths =glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_*_sigma0.3*_all.csv*') 
   paths = sorted(paths)
   #read DFs
   df_dic = read_dfs(paths,lambdac=True)
   cat_list=sorted(list(df_dic.keys()))

   comp_plot(df_dic,method_legend,filename=f'Comparison_SF',leg_title=leg_title,i=i,SF=cat_list,m_min=3,legend_loc=(1.02,0))
   comp_plot(df_dic,method_legend,filename=f'Comparison_Q',leg_title=leg_title,i=i,Q=cat_list,m_min=3,legend_loc=(1.02,0))
   comp_plot(df_dic,method_legend,filename=f'Comparison_SB',leg_title=leg_title,i=i,SB=cat_list,m_min=3,legend_loc=(1.02,0))

######### Schechter
leg_title='Eddington ratio distribution'
zees=[1.0,2.7]
# # Comparison between LX-M* relation of SF,Q,SB galaxies
for z in zees:
   method_legend=f'z = {z:.2f}\n'+r"M$_{\rm BH}$-M$_*$: Reines & Volonteri (2015)"
   i=reds_dic.get(z)
   paths =glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter/bs_perc_z{z}_*.csv*') 
   paths = sorted(paths)
   #read DFs
   df_dic = read_dfs(paths)
   cat_list=sorted(list(df_dic.keys()))

   comp_plot(df_dic,method_legend,filename=f'Comparison_SF_Schechter',leg_title=leg_title,i=i,SF=cat_list,m_min=3,legend_loc=(1.02,0))
   comp_plot(df_dic,method_legend,filename=f'Comparison_Q_Schechter',leg_title=leg_title,i=i,Q=cat_list,m_min=3,legend_loc=(1.02,0))
   comp_plot(df_dic,method_legend,filename=f'Comparison_SB_Schechter',leg_title=leg_title,i=i,SB=cat_list,m_min=3,legend_loc=(1.02,0))

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
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/bs_perc_z{z}_mean-2.25_sigma0.30*.csv') +\
      glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min/bs_perc_z{z}_mean-2.25_sigma0.30*_Schulze*.csv'))
print(paths)
#read DFs
df_dic = read_dfs(paths)
# remove common Eddington ratio distribution, which is the same for all:
dict_keys=list(df_dic.keys())
print(dict_keys)
match = SequenceMatcher(None, dict_keys[0], dict_keys[1]).find_longest_match(0, len(dict_keys[0]), 0, len(dict_keys[1]))
print(match)
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
paths = glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_mean-2.25_sigma0.30*_Georgakakis*.csv') + \
         glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min/bs_perc_z{z}_mean-2.25_sigma0.30*_Georgakakis*.csv')
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
paths = glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_mean-2.25_sigma0.30*_Man*.csv') + \
      glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min/bs_perc_z{z}_mean-2.25_sigma0.30*_Man*.csv')
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
paths = glob.glob(curr_dir+f'/Ros_plots/Duty_cycles/bs_perc_z{z}_mean-2.25_sigma0.30*_const0.2*.csv')+\
      glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min/bs_perc_z{z}_mean-2.25_sigma0.30*_const0.2*.csv')
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
#plt.ylim(2.1e-6,9e2)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/Test_lambda_min_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
plt.close(fig)


###################################################
######### Referee - Test lambda min ############### FLAT EDD DISTRIB
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
paths = sorted(glob.glob(curr_dir+f'/Ros_plots/R&V_Schechter/bs_perc_z1.0_alpha1.5_lambda0.0_lambdac-2.050.csv') +\
      glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min_flat_Schechter/bs_perc_z{z}_lambda0.00_alpha1.50*_Schulze*.csv'))
print(paths)
#read DFs
df_dic = read_dfs(paths)
# remove common Eddington ratio distribution, which is the same for all:
dict_keys=list(df_dic.keys())
print(dict_keys)
match = SequenceMatcher(None, dict_keys[0], dict_keys[1]).find_longest_match(0, len(dict_keys[0]), 0, len(dict_keys[1]))
print(match)
match=dict_keys[1][match.a: match.a + match.size+2]
new_keys= [a_string.replace(match, "") for a_string in dict_keys]
new_keys[0]=r'$\log(\lambda_{min})=$-4'
df_dic = dict(zip(new_keys, list(df_dic.values())))

q = generate_pie_markers(5)
markers_style=[q[0],q[1],q[2],q[3],q[4]]


method_legend="Edd ratio distr: "+match[:-2]+'\n'+r"M$_{\rm BH}$-M$_*$: Reines & Volonteri (2015)"+f"\nz={z}"
leg_title="Schulze"
comp_subplot(axs[0,0],df_dic,leg_title=leg_title,i=i,method_legend=method_legend)#,markers_style=markers_style

##########################################################
# top-right
# # Georgakakis17
paths = glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min_flat_Schechter/bs_perc_z{z}_lambda0.00_alpha1.50*_Georgakakis*.csv')
paths = sorted(paths)

#read DFs
df_dic = read_dfs(paths)
dict_keys=list(df_dic.keys())
new_keys= [a_string.replace(match, "") for a_string in dict_keys]
#new_keys[0]=r'$\log(\lambda_{min})=$ -4'
df_dic = dict(zip(new_keys, list(df_dic.values())))

leg_title='Georgakakis17'
comp_subplot(axs[0,1],df_dic,leg_title=leg_title,i=i)#,markers_style=markers_style
##########################################################
# bottom-left
# Man19
paths = glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min_flat_Schechter/bs_perc_z{z}_lambda0.00_alpha1.50*_Man*.csv')
paths = sorted(paths)

#read DFs
df_dic = read_dfs(paths)
dict_keys=list(df_dic.keys())
new_keys= [a_string.replace(match, "") for a_string in dict_keys]
#new_keys[0]=r'$\log(\lambda_{min})=$ -4'
df_dic = dict(zip(new_keys, list(df_dic.values())))

leg_title=r"Man19"
comp_subplot(axs[1,0],df_dic,leg_title=leg_title,i=i)#,markers_style=markers_style

##########################################################
# bottom-right
# constant=0.2
paths = glob.glob(curr_dir+f'/Ros_plots/Test_Edd_min_flat_Schechter/bs_perc_z{z}_lambda0.00_alpha1.50*_const0.2*.csv')
paths = sorted(paths)

#read DFs
df_dic = read_dfs(paths)
dict_keys=list(df_dic.keys())
new_keys= [a_string.replace(match, "") for a_string in dict_keys]
#new_keys[0]=r'$\log(\lambda_{min})=$ -4'
df_dic = dict(zip(new_keys, list(df_dic.values())))

leg_title=r'$const=0.2$'
comp_subplot(axs[1,1],df_dic,leg_title=leg_title,i=i)#,markers_style=markers_style
##########################################################
# invisible labels for creating space:
#axs[-1, 0].set_xlabel('.', color=(0, 0, 0, 0))
#axs[-1, 0].set_ylabel('.', color=(0, 0, 0, 0))
# common lables:
fig.text(0.5, 0.07, r'$\log$ <M$_*$> (M$_\odot$)', va='center', ha='center',size='x-large')
fig.text(0.08, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
#plt.ylim(2.1e-6,9e2)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/Test_lambda_min_z{z}_flatEdd.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
plt.close(fig)


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
df_dic = read_dfs(paths,lambdac=True)
cat_list=sorted(list(df_dic.keys()))
leg_title='Eddington ratio distribution'

SFR_LX(df_dic,leg_title, m_min = 4,filename='SFR_LX_SF',SF=cat_list)
SFR_LX(df_dic,leg_title, m_min = 4,filename='SFR_LX_Q',Q=cat_list)
SFR_LX(df_dic,leg_title, m_min = 4,filename='SFR_LX_SB',SB=cat_list)

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
      glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-3.0_sigma0.3*.csv*') +\
      glob.glob(curr_dir+f'/Ros_plots/Test_R&V_Gaussian_norm/bs_perc_z{z}_mean-2.25_sigma0.30*.csv*')
#paths = sorted(paths)
#read DFs
df_dic = read_dfs(paths,sigma=sigma,mean=mean,lambdac=True)
#print(df_dic.keys())
sb_list=['Gaussian $\\mu=-2.00$; $\\zeta_c=-1.90$']# 'Gaussian $\\mu=-2.25$; norm=8.0; $\\zeta_c=-2.15$'
q_list=['Gaussian $\\mu=-3.00$; $\\zeta_c=-2.90$']
sf_list=['Gaussian $\\mu=-2.25$; $\\zeta_c=-2.15$']#'Gaussian $\\mu=-2.25$; $\\zeta_c=-2.15$'

print('Gaussian $\\mu=-2.25$; norm=8.0; $\\zeta_c=-2.15$' in list(df_dic.keys()))

method_legend=f"z={z}"+"\n"+r"M$_{\rm BH}$-M$_*$: Reines & Volonteri (2015)"
comp_subplot(axs[0],df_dic,method_legend,leg_title=leg_title,i=i,Q=q_list,SB=sb_list,SF=sf_list,m_min=2,legend=False,markers_style=markers)
axs[0].set_xlabel('<M$_*$> (M$_\odot$)')

######################################################################
# second subplot
#print(df_dic.keys())
leg_title='Eddington ratio distribution'

SFR_LX_subplot(axs[1],df_dic,leg_title, m_min = 2,SB=sb_list,Q=q_list,SF=sf_list)
fig.text(0.07, 0.5, '<L$_X$> (2-10 keV) / $10^{42}$ (erg/s)', va='center', ha='center', rotation='vertical',size='x-large')
plt.ylim(2e-2,2.5e1)
plt.yscale('log')
plt.savefig(curr_dir+f'/Ros_plots/fig4.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
plt.close(fig)


#%%
##########################################################################################
############ Compare SF,Q,SB to Reines&Volonteri relation varying its normalization ############
##########################################################################################
# # Comparison between LX-M* relation of SF,Q,SB galaxies
z = 1.0
i=reds_dic.get(z)
paths =glob.glob(curr_dir+f'/Ros_plots/R&V_Gaussian/bs_perc_z{z}_mean-2.25_sigma0.3*.csv*') +\
      glob.glob(curr_dir+f'/Ros_plots/Test_R&V_Gaussian_norm/bs_perc_z{z}_mean-2.25_sigma0.3*.csv*') 
paths = sorted(paths)
#print(paths)
#read DFs
df_dic = read_dfs(paths)
SB_list=list(df_dic.keys())

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

SFR_LX(df_dic,leg_title, m_min = 4,filename='SFR_LX_SFQSB_norm',SB=sb_list,Q=q_list)
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
paths = glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/bs_perc_z{z}_lambda-2.35_alpha0.10*.csv') +\
        glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/Scatter_Mbh-M*_1dex/bs_perc_z{z}_lambda-2.35_alpha0.10*.csv') +\
        glob.glob(curr_dir+f'/Ros_plots/Standard_R&V/Scatter_Mbh-M*_1.5dex/bs_perc_z{z}_lambda-2.35_alpha0.10*.csv') 
#read DFs
keys=['0.55 dex (default)','1.0 dex','1.5 dex']
df_dic = read_dfs(paths,keys)

#method_legend=f"z={z}\nBH_mass: Shankar+16"
method_legend=f"z={z}\nBH_mass: Reines&Volonteri15"
leg_title='Scatter in Mbh-M* relation'

comp_plot(df_dic,method_legend,filename='Comparison_Scatter_Mbh-M*',leg_title=leg_title,i=i)
