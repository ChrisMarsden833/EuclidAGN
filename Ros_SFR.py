from AGNCatalogToolbox import main as agn
from AGNCatalogToolbox import Literature
from colossus.cosmology import cosmology
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dc_stat_think as dcst
import weightedstats as ws 
from scipy import interpolate

curr_dir=os.getcwd()

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

################################
# import data from IDL
from scipy.io import readsav
read_data = readsav('./IDL_data/vars_EuclidAGN_90.sav',verbose=True)

data={}
for key, val in read_data.items():
    data[key]=np.copy(val)
    data[key][data[key] == 0.] = np.nan
#print(data.keys())

################################
# my cosmology
params = {'flat': True, 'H0': 70., 'Om0': 0.3, 'Ob0': 0.0486, 'sigma8':0.8159, 'ns':0.9667}
cosmology.addCosmology('Carraro+20', params)

cosmo = 'Carraro+20'
cosmology = cosmology.setCosmology(cosmo)
volume = 200**3 # Mpc?

### Define parameters for code to work
slope=None # to be used with Davis18
norm=None # to be used with R&V
suffix=''

################################
# import simulation parameters
sub_dir='Sahu_Gaussian/' # z=1: pars1. z=2.7 :pars2. z=0.45:pars3 
#sub_dir='R&V_Gaussian/' # z=1: pars1. z=2.7 :pars2. z=0.45:pars3 
sub_dir= 'R&V_Schechter/' # z=1: pars1. 
sub_dir= 'Scaling_rels_bestLF/'
sub_dir= 'Scaling_rels_restr/'
sub_dir= 'Duty_cycles/'
sub_dir= 'Standard/'
sub_dir='Test_R&V_Gaussian_SFQSB/' # up to pars11
sub_dir= 'Scaling_rels/'
sub_dir= 'Duty_cycles/'
sub_dir= 'Davis_slope/'
sys.path.append(curr_dir+'/Ros_plots/'+sub_dir)

from pars5 import *

if methods['edd_ratio']=='Gaussian':
   lambda_z=sigma_z
   alpha_z=mu_z

################################
## Generate universe ##
################################
gals = pd.DataFrame()

# Halos
gals['halos'] = agn.generate_semi_analytic_halo_catalogue(volume, [10, 16, 0.1], z, params.get('H0')/100)

# Galaxies
gals['stellar_mass'] = agn.halo_mass_to_stellar_mass(gals.halos, z, formula=methods['halo_to_stars'])

# limit mass range to validity of scaling relation
if M_inf > 0:
    gals=gals[gals['stellar_mass'] >= M_inf]
if M_sup > 0:
    M_sup=np.min([12,M_sup])
else:
    M_sup=12
gals=gals[gals['stellar_mass'] <= M_sup]
print(gals.stellar_mass.min(),gals.stellar_mass.max())

# BH
if slope is not None:
   if methods['BH_mass_method'] != 'Davis18': print('Warning! slope parameter is meant to be used with Davis+18 relation only')
   gals['black_hole_mass'] = agn.stellar_mass_to_black_hole_mass(gals.stellar_mass, method = methods['BH_mass_method'], 
                                                                  scatter = methods['BH_mass_scatter'],slope=slope)
elif norm is not None:
   if methods['BH_mass_method'] != 'Reines&Volonteri15': print('Warning! slope parameter is meant to be used with Reines&Volonteri15 relation only')
   gals['black_hole_mass'] = agn.stellar_mass_to_black_hole_mass(gals.stellar_mass, method = methods['BH_mass_method'], 
                                                                  scatter = methods['BH_mass_scatter'],norm=norm)
else:
   gals['black_hole_mass'] = agn.stellar_mass_to_black_hole_mass(gals.stellar_mass, method = methods['BH_mass_method'], 
                                                                  scatter = methods['BH_mass_scatter'])

# Duty cycles
gals['duty_cycle'] = agn.to_duty_cycle(methods['duty_cycle'], gals.stellar_mass, gals.black_hole_mass, z)

gals['luminosity'], lambda_char, _ = agn.black_hole_mass_to_luminosity(
                                          gals.black_hole_mass, 
                                          z, methods['edd_ratio'],
                                          bol_corr=methods['bol_corr'], parameter1=lambda_z, parameter2=alpha_z)
                                          #bol_corr=methods['bol_corr'], parameter1=sigma_z, parameter2=mu_z)
"""
###############
# plot XLF
plt.figure()
# XLF Data
print('z=',z)
XLF = Literature.XLFData(z)
mXLF_data = XLF.get_miyaji2015()
plt.plot(mXLF_data.x, mXLF_data.y, 'o', label = "Miyaji")
uXLF_data = XLF.get_ueda14(np.arange(42, 46, 0.1))
plt.plot(uXLF_data.x, uXLF_data.y, ':', label = "Ueda")

plt.plot(10**XLF_plotting_data.x, 10**XLF_plotting_data.y, label = 'Simulation')# finish plot
plt.xlabel(r'$L_x\;[erg\;s^{-1}]$')
plt.ylabel(r'$d\phi /d(log\;L_x)\;[Mpc^{-3}]$')
plt.loglog()
plt.legend()
if methods['edd_ratio']=="Schechter":
   title_str=r'XLF, U={}, $\alpha$={:.2f}, $\lambda$={:.2f}, z={}'.format(methods['duty_cycle'],alpha_z,lambda_z,z)
   file_name=curr_dir+'/Ros_plots/'+sub_dir+f'XLF_z{z}_alpha{alpha_z:.2f}_lambda{lambda_z:.2f}.pdf'
elif methods['edd_ratio']=="Gaussian":
   title_str=r'XLF, U={}, $\mu$={}, $\sigma$={}, z={}'.format(methods['duty_cycle'],mu_z,sigma_z,z)
   file_name=curr_dir+'/Ros_plots/'+sub_dir+f'XLF_z{z}_mean{mu_z:.2f}_sigma{sigma_z:.2f}.pdf'
plt.title(title_str)
plt.savefig(file_name, format = 'pdf', bbox_inches = 'tight',transparent=True)
#plt.show()
#################
# do squared distance from Miyaji
XLF_plot_int=interpolate.interp1d(10**XLF_plotting_data.x, 10**XLF_plotting_data.y)
good_ones=np.logical_and((mXLF_data.x>=10**XLF_plotting_data.x[0]), (mXLF_data.x<=10**XLF_plotting_data.x[-1]))
mXLF_x=mXLF_data.x[good_ones]
XLF_sim=XLF_plot_int(mXLF_x)
S=np.sum((np.log10(XLF_sim)-np.log10(mXLF_data.y[good_ones]))**2)
"""
#Save to file the characteristic lambda
print(f'lambda{lambda_z:.2f}_alpha{alpha_z:.2f}:\tlambda_char={lambda_char:.2f}')#, Squares sum={S:.2e}
if methods['edd_ratio']=="Schechter":
   append_new_line(curr_dir+'/Ros_plots/'+sub_dir+'Squared_test.txt', f'z={z}, lambda={lambda_z:.2f}, alpha={alpha_z:.2f}:\tlambda_char={lambda_char:.2f}')#, Squares sum={S:.2e}
elif methods['edd_ratio']=="Gaussian":
   append_new_line(curr_dir+'/Ros_plots/'+sub_dir+'Squared_test.txt', f'z={z}, mean={alpha_z:.2f}, sigma={lambda_z:.2f}:\tlambda_char={lambda_char:.2f}')#, Squares sum={S:.2e}

#gals['nh'] = agn.luminosity_to_nh(gals.luminosity, z)
#gals['agn_type'] = agn.nh_to_type(gals.nh)

gals['SFR'] = agn.SFR(z,gals.stellar_mass,methods['SFR'])
gals['lx/SFR'] = (gals.luminosity-42)-gals.SFR

gals['SFR_Q'] = agn.SFR_Q(z,gals.stellar_mass)
gals['SFR_SB'] = agn.SFR_SB(z,gals.stellar_mass)

################################
# grouping in mass bins - log units
#grouped_gals = gals[['stellar_mass','luminosity','SFR','lx/SFR','duty_cycle']].groupby(pd.cut(gals.stellar_mass, np.append(np.arange(5, 11.5, 0.5),12.))).quantile([0.05,0.1585,0.5,0.8415,0.95]).unstack(level=1)

# converting to linear units
gals_lin=pd.DataFrame()
gals_lin['stellar_mass'] = gals['stellar_mass']
gals_lin['duty_cycle'] = gals['duty_cycle']
gals_lin['luminosity']= 10**(gals.luminosity-42)
gals_lin[['SFR','lx/SFR']]=10**gals[['SFR','lx/SFR']]
gals_lin['SFR_Q'] = 10**(gals.SFR_Q)
gals_lin['SFR_SB'] = 10**(gals.SFR_SB)

# grouping linear table
grouped_lin = gals_lin[['stellar_mass','luminosity','SFR','SFR_Q','SFR_SB','lx/SFR','duty_cycle']].groupby(pd.cut(gals.stellar_mass, np.append(np.arange(5, 11.5, 0.5),12.5))).quantile([0.05,0.1585,0.5,0.8415,0.95]).unstack(level=1)
# limit to logM>9
ggals_lin=grouped_lin[grouped_lin['stellar_mass',0.5] > 9]
grouped_lin.index.rename('mass_range',inplace=True)

################################
## Bootstrapping ##
################################
M_min=np.max([9,M_inf])

# create dataframe for bootstrapping
gals_highM=gals_lin.copy()[gals_lin.stellar_mass > M_min]
grouped_linear = gals_highM[['stellar_mass','luminosity','SFR','SFR_Q','SFR_SB','lx/SFR','duty_cycle']].groupby(pd.cut(gals_highM.stellar_mass, np.append(np.arange(M_min, 11.5, 0.5),12.5)))#.quantile([0.05,0.1585,0.5,0.8415,0.95]).unstack(level=1)

# create dataframe of bootstraped linear varibles
gals_bs=pd.DataFrame()
func=np.median
gals_bs['SFR'] = grouped_linear.SFR.apply(lambda x: dcst.draw_bs_reps(x, func, size=500))
gals_bs['SFR_Q'] = grouped_linear.SFR_Q.apply(lambda x: dcst.draw_bs_reps(x, func, size=500))
gals_bs['SFR_SB'] = grouped_linear.SFR_SB.apply(lambda x: dcst.draw_bs_reps(x, func, size=500))

func=ws.weighted_median
def weighted_bootsrap(x,y):
   # draw_bs_reps applies ws.weighted_median(x_bs_i, weights=y)
   return dcst.draw_bs_reps(x, func, size=500,args=(y,))
gals_bs['luminosity'] = grouped_linear.apply(lambda x: weighted_bootsrap(x.luminosity,x.duty_cycle))
#gals_bs['luminosity'] = grouped_linear.luminosity.apply(lambda x: dcst.draw_bs_reps(x, func, size=500)) #with median
gals_bs.head()

# create dataframe with percentiles of the bootstrapped distribution
bs_perc=ggals_lin['stellar_mass']
old_idx = bs_perc.columns.to_frame()
perc_colnames=bs_perc.columns
old_idx.insert(0, '', 'stellar_mass')
bs_perc.columns = pd.MultiIndex.from_frame(old_idx)

bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row,[0.05,0.1585,0.5,0.8415,0.95]) for row in gals_bs['SFR']]), 
                                  index=bs_perc.index, columns=pd.MultiIndex.from_product([['SFR'],perc_colnames])))
bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row,[0.05,0.1585,0.5,0.8415,0.95]) for row in gals_bs['SFR_Q']]), 
                                  index=bs_perc.index, columns=pd.MultiIndex.from_product([['SFR_Q'],perc_colnames])))
bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row,[0.05,0.1585,0.5,0.8415,0.95]) for row in gals_bs['SFR_SB']]), 
                                  index=bs_perc.index, columns=pd.MultiIndex.from_product([['SFR_SB'],perc_colnames])))
bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row,[0.05,0.1585,0.5,0.8415,0.95]) for row in gals_bs['luminosity']]), 
                                  index=bs_perc.index, columns=pd.MultiIndex.from_product([['luminosity'],perc_colnames])))
bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row['luminosity']/row['SFR'],[0.05,0.1585,0.5,0.8415,0.95]) for i,row in gals_bs.iterrows()]), 
                                  index=bs_perc.index, columns=pd.MultiIndex.from_product([['lx_SFR'],perc_colnames])))
#print(bs_perc)

# save dataframe to file and add to dictionary for use
if methods['edd_ratio']=="Schechter":
  bs_perc.to_csv(curr_dir+'/Ros_plots/'+sub_dir+f'bs_perc_z{z}_lambda{lambda_z:.2f}_alpha{alpha_z:.2f}'+suffix+'.csv')
elif methods['edd_ratio']=="Gaussian":
  bs_perc.to_csv(curr_dir+'/Ros_plots/'+sub_dir+f'bs_perc_z{z}_mean{alpha_z:.2f}_sigma{lambda_z:.2f}'+suffix+'.csv')

################################
## Plot ##
################################
# Generic definitions
i=index

# errorbars of bootstrapped simulation points
xerr=np.array([bs_perc['SFR',0.5] - bs_perc['SFR',0.05], 
              bs_perc['SFR',0.95] - bs_perc['SFR',0.5]])
yerr=np.array([bs_perc['luminosity',0.5] - bs_perc['luminosity',0.05], 
               bs_perc['luminosity',0.95] - bs_perc['luminosity',0.5]])

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
    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
    ls=['--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]

    # "real" datapoints
    ax.scatter(data['m_ave'][0,2:,i], data['l_ave'][0,2:,i], edgecolors='Black', marker="s",label='Carraro et al. (2020)')
    ax.errorbar(data['m_ave'][0,2:,i], data['l_ave'][0,2:,i],
                    yerr=np.array([data['l_ave'][0,2:,i] - data['l_ave'][2,2:,i], 
                        data['l_ave'][1,2:,i] - data['l_ave'][0,2:,i]]),
                    linestyle='solid', zorder=0)

    # simulated datasets
    for j,(s,df) in enumerate(df_dic.items()):
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
    plt.savefig(curr_dir+'/Ros_plots/'+sub_dir+filename+f'_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) ;
    return

################################
# Plot SFR vs LX
# define parameters for datapoints' colors and colorbar
_min = np.minimum(np.nanmin(bs_perc['stellar_mass',0.5]),np.nanmin(data['m_ave'][0,2:,:]))
_max = np.maximum(np.nanmax(bs_perc['stellar_mass',0.5]),np.nanmax(data['m_ave'][0,2:,:]))

fig = plt.figure(figsize=[12, 8])
#plt.rcParams['figure.figsize'] = [12, 8]

# simulated datapoints
plt.scatter(bs_perc['SFR',0.5],bs_perc['luminosity',0.5], vmin = _min, vmax = _max, edgecolors='Black',
            c=bs_perc['stellar_mass',0.5] , s=bs_perc['stellar_mass',0.5]*10, label='Simulation')
plt.errorbar(bs_perc['SFR',0.5],bs_perc['luminosity',0.5],
                xerr=xerr, yerr=yerr, linestyle='--', zorder=0)

# "real" datapoints
sc=plt.scatter(data['sfr_ave'][0,2:,i], data['l_ave'][0,2:,i], vmin = _min, vmax = _max, edgecolors='Black',
            c=data['m_ave'][0,2:,0], s=data['m_ave'][0,2:,0]*10, marker="s",label='Data')
plt.errorbar(data['sfr_ave'][0,2:,i], data['l_ave'][0,2:,i],
                xerr=[data['sfr_ave'][0,2:,i]-data['sfr_ave'][2,2:,i],
                    data['sfr_ave'][1,2:,i]-data['sfr_ave'][0,2:,i]],
                yerr=np.array([data['l_ave'][0,2:,i] - data['l_ave'][2,2:,i], 
                    data['l_ave'][1,2:,i] - data['l_ave'][0,2:,i]]),
                linestyle='-', zorder=0)

# colorbar, labels, legend, etc
plt.colorbar(sc).set_label('Stellar mass (M$_\odot$)')
text_pars=dict(horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor='gray', alpha=0.5))
plt.text(0.137, 0.78, f'z = {z:.1f}', transform=fig.transFigure, **text_pars)
plt.text(0.137, 0.74, f"Halo to M* = {methods['halo_to_stars']}\nEddington ratio = {methods['edd_ratio']}\nDuty Cycle = {methods['duty_cycle']}\nBH_mass = {methods['BH_mass_method']}\nBol corr = {methods['bol_corr']}", transform=fig.transFigure, **text_pars)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('SFR (M$_\odot$/yr)')
plt.ylabel('L$_X$ (2-10 keV) / $10^{42}$ (erg/s)')
plt.legend(loc='upper left');
plt.savefig(curr_dir+'/Ros_plots/'+sub_dir+f'SFvsLX_z{z}'+suffix+'.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 