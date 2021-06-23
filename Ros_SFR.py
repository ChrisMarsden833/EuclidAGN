from AGNCatalogToolbox import main as agn
from AGNCatalogToolbox import Literature
from AGNCatalogToolbox.Ros_utilities import *
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
#from scipy.stats import binned_statistic
import scipy as sp
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

curr_dir=os.getcwd()

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
lambda_min=-4
slope=None # to be used with R&V
norm=None # to be used with R&V
suffix=''

################################
# import simulation parameters
sub_dir='Sahu_Gaussian/' # z=1: pars1. z=2.7 :pars2. z=0.45:pars3 
#sub_dir='R&V_Gaussian/' # z=1: pars1. z=2.7 :pars2. z=0.45:pars3 
sub_dir= 'R&V_Schechter/' # z=1: pars1. 
sub_dir= 'Scaling_rels_bestLF/'
sub_dir= 'Standard/Test_marconi/'
sub_dir= 'Standard_extended/'
sub_dir= 'Test_dutycycle_scatter/'
sub_dir= 'Test_dutycycle_onoff/'
sub_dir= 'Standard/Scatter_Mbh-M*_4*/'
sub_dir= 'Scaling_rels_restr/'
sub_dir= 'Scaling_rels/'
sub_dir= 'Standard/Scatter_Mh-M*_0.3/'
sub_dir= 'Standard_R&V/Scatter_Mh-M*_0.6/'
sub_dir= 'Standard_R&V/Scatter_Mbh-M*_4*/'
sub_dir='Test_R&V_Gaussian_SFQSB/' # up to pars11
sub_dir='Test_R&V_Gaussian_slope2_norm/' # up to pars11
sub_dir= 'Davis_slope/'
sub_dir= 'Scaling_rels/'
sub_dir= 'Test_Edd_min/' # up to pars16
sub_dir='Test_R&V_Gaussian_norm/' # up to pars11
sub_dir= 'Duty_cycles/'
sub_dir= 'Standard_R&V/'
sys.path.append(curr_dir+'/Ros_plots/'+sub_dir)

from pars1 import *

if methods['edd_ratio']=='Gaussian':
   lambda_z=sigma_z
   alpha_z=mu_z

################################
## Generate universe ##
################################
gals = pd.DataFrame()

# Halos
gals['halos'] = agn.generate_semi_analytic_halo_catalogue(volume, [10, 16, 0.1], z, params.get('H0')/100)

#plt.hist(gals['halos'])
#plt.ylabel('halos')
#plt.show()

# Galaxies
gals['stellar_mass'] = agn.halo_mass_to_stellar_mass(gals.halos, z, formula=methods['halo_to_stars'])

#plt.hist(gals['stellar_mass'])
#plt.ylabel('stellar_mass')
#plt.show()


# limit mass range to validity of scaling relation
if M_inf > 0:
    gals=gals[gals['stellar_mass'] >= M_inf]
if M_sup > 0:
    M_sup=np.min([12,M_sup])
else:
    M_sup=12
gals=gals[gals['stellar_mass'] <= M_sup]
print(gals.stellar_mass.min(),gals.stellar_mass.max())

#plt.hist(gals['stellar_mass'])
#plt.ylabel('stellar_mass >10')
#plt.show()

# BH
if (slope is not None) and (norm is not None):
   if methods['BH_mass_method'] != 'Reines&Volonteri15': print('Warning! slope parameter is meant to be used with Reines&Volonteri15 relation only')
   gals['black_hole_mass'] = agn.stellar_mass_to_black_hole_mass(gals.stellar_mass, method = methods['BH_mass_method'], 
                                                                  scatter = methods['BH_mass_scatter'],slope=slope,norm=norm)
elif slope is not None:
   if methods['BH_mass_method'] != 'Reines&Volonteri15': print('Warning! slope parameter is meant to be used with Reines&Volonteri15 relation only')
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

#plt.hist(gals['duty_cycle'])
#plt.ylabel('duty_cycle')
#plt.show()

gals['luminosity'], lambda_char, gals['edd_ratio'] = agn.black_hole_mass_to_luminosity(
                                          gals.black_hole_mass, 
                                          z, methods['edd_ratio'],
                                          bol_corr=methods['bol_corr'], parameter1=lambda_z, parameter2=alpha_z,lambda_min=lambda_min)
                                          #bol_corr=methods['bol_corr'], parameter1=sigma_z, parameter2=mu_z)

#plt.hist(gals['luminosity'])
#plt.ylabel('luminosity')
#plt.show()
#%%
###############
# determine XLF
step = 0.2
bins = np.arange(42., 46., step)
#lum_bins = binned_statistic(gals['luminosity'], gals['duty_cycle'], 'sum', bins=bins)[0]
lum_bins = sp.stats.binned_statistic(gals['luminosity'], gals['duty_cycle'], 'sum', bins=bins)[0]
lum_func = (lum_bins/volume)/step
bins1 = bins[0:-1][lum_func > 0]
lum1 = np.log10(lum_func[lum_func > 0])

from AGNCatalogToolbox import Utillity as utl
xlf_plotting_data = utl.PlottingData(bins[0:-1][lum_func > 0], np.log10(lum_func[lum_func > 0]))

# plot XLF
plt.figure()
# XLF Data
print('z=',z)
XLF = Literature.XLFData(z)
mXLF_data = XLF.get_miyaji2015()
plt.plot(mXLF_data.x, mXLF_data.y, 'o', label = "Miyaji")
uXLF_data = XLF.get_ueda14(np.arange(42, 46, 0.1))
plt.plot(uXLF_data.x, uXLF_data.y, ':', label = "Ueda")

plt.plot(10**xlf_plotting_data.x, 10**xlf_plotting_data.y, label = 'Simulation')# finish plot
plt.xlabel(r'$L_x\;[erg\;s^{-1}]$')
plt.ylabel(r'$d\phi /d(log\;L_x)\;[Mpc^{-3}]$')
plt.loglog()
plt.legend()
if methods['edd_ratio']=="Schechter":
   title_str=r'XLF, U={}, $\alpha$={:.2f}, $\lambda$={:.2f}, z={}'.format(methods['duty_cycle'],alpha_z,lambda_z,z)
   file_name=curr_dir+'/Ros_plots/'+sub_dir+f'XLF_z{z}_alpha{alpha_z:.2f}_lambda{lambda_z:.2f}'+suffix+'.pdf'
elif methods['edd_ratio']=="Gaussian":
   title_str=r'XLF, U={}, $\mu$={}, $\sigma$={}, z={}'.format(methods['duty_cycle'],mu_z,sigma_z,z)
   file_name=curr_dir+'/Ros_plots/'+sub_dir+f'XLF_z{z}_mean{mu_z:.2f}_sigma{sigma_z:.2f}'+suffix+'.pdf'
plt.title(title_str)
print(file_name)
plt.savefig(file_name, format = 'pdf', bbox_inches = 'tight',transparent=True)
#plt.show()
#################
# do squared distance from Miyaji
XLF_plot_int=interpolate.interp1d(10**xlf_plotting_data.x, 10**xlf_plotting_data.y)
good_ones=np.logical_and((mXLF_data.x>=lum_bins[0]), (mXLF_data.x<=lum_func[-1]))
mXLF_x=mXLF_data.x[good_ones]
XLF_sim=XLF_plot_int(mXLF_x)
S=np.sum((np.log10(XLF_sim)-np.log10(mXLF_data.y[good_ones]))**2)
#%%
#Save to file the characteristic lambda
print(f'lambda{lambda_z:.2f}_alpha{alpha_z:.2f}:\tlambda_char={lambda_char:.3f}')#, Squares sum={S:.2e}
if methods['edd_ratio']=="Schechter":
   append_new_line(curr_dir+'/Ros_plots/'+sub_dir+'Squared_test.txt', f'z={z}, lambda={lambda_z:.2f}, alpha={alpha_z:.2f}:\tlambda_char={lambda_char:.3f}')#, Squares sum={S:.2e}
elif methods['edd_ratio']=="Gaussian":
   append_new_line(curr_dir+'/Ros_plots/'+sub_dir+'Squared_test.txt', f'z={z}, mean={alpha_z:.2f}, sigma={lambda_z:.2f}:\tlambda_char={lambda_char:.3f}')#, Squares sum={S:.2e}

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
#gals_lin['is_active'] = is_active(gals_lin.duty_cycle)
gals_lin['luminosity']= 10**(gals.luminosity-42)
gals_lin['weighted_luminosity']= gals_lin['luminosity']*gals['duty_cycle']
#gals_lin['luminosity2']= gals_lin['luminosity']*gals_lin.is_active
#gals_lin['luminosity2'].loc[gals_lin['luminosity2']==0]=1e-6
gals_lin[['SFR','lx/SFR']]=10**gals[['SFR','lx/SFR']]
gals_lin['SFR_Q'] = 10**(gals.SFR_Q)
gals_lin['SFR_SB'] = 10**(gals.SFR_SB)
gals_lin['black_hole_mass'] = gals['black_hole_mass']
gals_lin['edd_ratio'] = gals['edd_ratio']
print(gals_lin[['SFR','SFR_Q','SFR_SB']])

dfs_dict=dict(SF='',Q='_Q',SB='_SB')
bs_perc=pd.DataFrame()

for key, str in dfs_dict.items():
   sfr_str='SFR'+str
   mstar_str='stellar_mass'+str
   lum_str='luminosity'+str
   lambda_str='lambda_ave'+str

   # remove galaxies with lx < lx_XRB, as of Lehmer+16, from both dfs
   #gals_lin['lx_bin']=10**(29.37+2.03*np.log10(1+z)+gals_lin.stellar_mass-42)+10**(39.28+1.31*np.log10(1+z)+gals[sfr_str]-42)
   #α0(1 + z)γ M∗ + β0(1 + z)δSFR, (3)
   #with logα0 = 29.37, logβ0 = 39.28, γ = 2.03, and δ = 1.31.
   #to_drop=gals_lin['luminosity']<gals_lin['lx_bin']
   #print(f'Fraction galaxies with LX<LX_bin: {to_drop.sum()/(gals_lin.shape[0]):.3f}')

   gals_lin_tmp=gals_lin.copy()#[gals_lin['luminosity']>gals_lin['lx_bin']]
   gals_lin_tmp=gals_lin_tmp[gals_lin_tmp['stellar_mass']>9]

   ################################
   ## Bootstrapping ##
   ################################
   M_min=np.max([9,M_inf])

   # create dataframe for bootstrapping - M_min gets rounded to the lower .5 number
   gals_highM=gals_lin_tmp.copy()[gals_lin_tmp.stellar_mass > M_min]
   grouped_linear = gals_highM[['stellar_mass','luminosity','SFR','SFR_Q','SFR_SB','lx/SFR','duty_cycle','edd_ratio','weighted_luminosity']].groupby(pd.cut(gals_highM.stellar_mass, np.append(np.arange(np.floor(M_min*2)/2, 11.5, 0.5),12.5)))#.quantile([0.05,0.1585,0.5,0.8415,0.95]).unstack(level=1)
   #print('Statistics for mock sample:')
   #print(gals_highM[['stellar_mass','black_hole_mass','duty_cycle']].describe(percentiles=[.01,.05,.25, .5, .75,.95,.99]))

   # plot masses
   stellar_mass=np.linspace(gals_highM['stellar_mass'].min(),gals_highM['stellar_mass'].max())
   plt.figure()
   plt.scatter(gals_highM['stellar_mass'],gals_highM['black_hole_mass'],label='Mock galaxies',alpha=0.2,s=0.2)
   plt.plot(stellar_mass,7.574 + 1.946 * (stellar_mass - 11.) - 0.306 * (stellar_mass - 11.)**2.- 0.011 * (stellar_mass - 11.)**3,label='Shankar+16',c='Orange')
   plt.ylabel('black_hole_mass')
   plt.xlabel('stellar_mass')
   #plt.xlim(10,gals_highM['stellar_mass'].max())
   #plt.ylim(3.4,gals_highM['black_hole_mass'].max()+0.2)
   plt.legend()
   file_name=curr_dir+'/Ros_plots/'+sub_dir+f'masses_rel'+suffix+'.pdf'
   plt.savefig(file_name, format = 'pdf', bbox_inches = 'tight',transparent=True)

   # plot duty cycles distributions
   fig,axs = plt.subplots(3,2,figsize=[15, 15])
   #--------
   trans = axs[0,0].get_xaxis_transform()
   #grouped_linear.duty_cycle.hist(ax=axs[0],alpha=0.8,legend=True,bins=np.logspace(np.log10(0.0008),np.log10(0.5), 50),histtype = 'step')
   #axs[0].set_xlabel('duty cycle')
   grouped_linear.edd_ratio.hist(ax=axs[0,0],alpha=0.8,legend=True,bins=np.linspace(-4,1, 50),histtype = 'step',density=True,grid=False)
   for col, (mass, value) in zip(mcolors.TABLEAU_COLORS,grouped_linear.edd_ratio.mean().items()):
      axs[0,0].axvline(value,alpha=0.8,color=col,ls='--')
      axs[0,0].text(value,0.05,f'Mean {mass}: {value:.2f}',rotation=90, transform=trans,alpha=0.6)
   axs[0,0].set_xlabel('Eddington ratio')
   axs[0,0].set_yscale('log')
   axs[0,0].xaxis.set_minor_locator(AutoMinorLocator())
   #axs[0].set_xscale('log')
   #axs[0].axvline(grouped_linear.duty_cycle.median())
   #--------
   """
   ax2=fig.add_subplot(121, label="2", frame_on=False)
   grouped_linear.duty_cycle.hist(ax=ax2,alpha=0.8,legend=True,bins=np.logspace(np.log10(0.0008),np.log10(0.5), 50),histtype = 'step',linestyle=('dashed'))
   #ax2.scatter(x_values2, y_values2, color="C1")
   ax2.xaxis.tick_top()
   ax2.yaxis.tick_right()
   ax2.set_xlabel(f"Duty cycle, {methods['duty_cycle']}", color="darkblue") 
   ax2.set_ylabel('', color="darkblue")       
   ax2.xaxis.set_label_position('top') 
   ax2.yaxis.set_label_position('right') 
   ax2.set_xscale('log')
   ax2.tick_params(axis='x', colors="darkblue")
   ax2.tick_params(axis='y', colors="darkblue")
   """
   #--------
   grouped_linear['luminosity'].hist(ax=axs[0,1],bins=np.logspace(np.log10(gals_highM['luminosity'].min()),np.log10(gals_highM['luminosity'].max()), 50) ,legend=True,histtype = 'step',grid=False)
   #grouped_linear.weighted_luminosity.hist(ax=axs[1],bins=np.logspace(np.log10(1e-6),np.log10(2e3), 50) ,alpha=0.7,legend=True,histtype = 'step')
   trans = axs[0,1].get_xaxis_transform()
   #axs[1].vlines(grouped_linear.duty_cycle.median())
   for col, (mass, value) in zip(mcolors.TABLEAU_COLORS,grouped_linear['luminosity'].median().items()):
      axs[0,1].axvline(value,alpha=0.6,color=col,ls='--')
      axs[0,1].text(value,0.5,f'Median {mass}: {np.log10(value)+42:.2f}',rotation=90, transform=trans,alpha=0.6)
   for col, (mass, value)  in zip(mcolors.TABLEAU_COLORS,grouped_linear['luminosity'].mean().items()):
      axs[0,1].axvline(value,alpha=0.8,color=col)
      axs[0,1].text(value,0.05,f'Mean {mass}: {np.log10(value)+42:.2f}',rotation=90, transform=trans)
   #print('Lum median', np.log10(grouped_linear.luminosity.median()*10**(42)))
   #print('weighted Lum median', np.log10(grouped_linear.weighted_luminosity.median()*10**(42)))
   #grouped_linear.luminosity2.hist(ax=axs[1],alpha=0.4,bins=np.logspace(np.log10(1e-6),np.log10(2e3), 50),legend=True)
   axs[0,1].set_yscale('log')
   axs[0,1].set_xscale('log')
   axs[0,1].set_xlabel('LX / 1e42 erg/s')
   #--------
   grouped_linear.duty_cycle.hist(ax=axs[1,0],alpha=0.8,legend=True,bins=np.logspace(np.log10(0.0008),np.log10(0.5), 50),histtype = 'step',linestyle=('dashed'),grid=False)
   axs[1,0].set_xscale('log')
   axs[1,0].set_yscale('log')
   axs[1,0].set_ylim(bottom=1e-1)
   axs[1,0].set_xlabel(f"Duty cycle, {methods['duty_cycle']}")
   axs[1,0].legend(loc='upper left')
   #--------
   axs[1,1].get_shared_x_axes().join(axs[1,1], axs[0,1])
   trans = axs[1,1].get_xaxis_transform()
   grouped_linear.weighted_luminosity.hist(ax=axs[1,1],bins=np.logspace(np.log10(gals_highM['weighted_luminosity'].min()),np.log10(gals_highM['weighted_luminosity'].max()), 70) ,alpha=0.7,legend=True,histtype = 'step',grid=False)
   wmean=grouped_linear.weighted_luminosity.sum()/grouped_linear.duty_cycle.sum()
   #print(wmean)
   for col, (mass, value)  in zip(mcolors.TABLEAU_COLORS,grouped_linear.weighted_luminosity.mean().items()):
      axs[1,1].axvline(value,alpha=0.6,color=col,ls='--')
      axs[1,1].text(value,0.05,f'Mean {mass}: {np.log10(value)+42:.2f}',rotation=90, transform=trans,alpha=0.6)
   for col, (mass, value)  in zip(mcolors.TABLEAU_COLORS,wmean.items()):
      axs[1,1].axvline(value,alpha=0.8,color=col)
      axs[1,1].text(value,0.5,f'Weighted mean {mass}: {np.log10(value)+42:.2f}',rotation=90, transform=trans)
   axs[1,1].set_yscale('log')
   axs[1,1].set_xscale('log')
   axs[1,1].set_xlabel('LX / 1e42 * Duty_cycle [erg/s]')
   #--------

   # create dataframe of bootstraped linear varibles
   gals_bs=pd.DataFrame()
   func=np.median
   gals_bs[sfr_str] = grouped_linear[sfr_str].apply(lambda x: dcst.draw_bs_reps(x, func, size=500))
   #gals_bs['unweighted_luminosity'] = grouped_linear.luminosity.apply(lambda x: dcst.draw_bs_reps(x, func, size=500)) #with median

   gals_bs['weighted_median'] = grouped_linear.apply(lambda x: my_draw_bs_reps(x.luminosity.values,x.duty_cycle.values, size=500))
   #--------
   trans = axs[2,0].get_xaxis_transform()
   for col, (index, item) in zip(mcolors.TABLEAU_COLORS,gals_bs['weighted_median'].items()):
      axs[2,0].hist(item,bins=np.logspace(np.log10(gals_highM['weighted_luminosity'].min()),np.log10(gals_highM['weighted_luminosity'].max()), 50),histtype = 'step', label=index)
      median_val=np.median(item)
      axs[2,0].axvline(median_val,color=col,alpha=0.6,ls='--')
      axs[2,0].text(median_val,0.5,f'Median {index}: {np.log10(median_val)+42:.2f}',rotation=90, transform=trans,alpha=0.6)
   axs[2,0].get_shared_x_axes().join(axs[2,0], axs[0,1])
   axs[2,0].set_xscale('log')
   axs[2,0].set_yscale('log')
   axs[2,0].legend()
   axs[2,0].set_xlabel('Bootstrapped weighted median luminosities')
   #--------

   gals_bs[lum_str] = grouped_linear.apply(lambda x: my_draw_bs_reps(x.luminosity.values,x.duty_cycle.values, size=500,type='mean'))
   #print(gals_bs.head())
   #gals_bs[lum_str].hist(ax=axs[2,1],alpha=0.8,legend=True,bins=np.logspace(np.log10(0.0008),np.log10(0.5), 50),histtype = 'step')
   #--------
   for col, (index, item) in zip(mcolors.TABLEAU_COLORS,gals_bs[lum_str].items()):
      axs[2,1].hist(item,bins=np.logspace(np.log10(1e-6),np.log10(2e3), 50),histtype = 'step', label=index)
      axs[2,1].axvline(np.median(item),color=col,alpha=0.6,ls='--')
   axs[2,1].get_shared_x_axes().join(axs[2,1], axs[0,1])
   axs[2,1].set_xscale('log')
   axs[2,1].set_yscale('log')
   axs[2,1].legend()
   axs[2,1].set_xlabel('Bootstrapped weighted mean luminosities')
   #--------
   file_name=curr_dir+'/Ros_plots/'+sub_dir+f'duty_cycle_distrib'+suffix+f'_{key}.pdf'
   plt.savefig(file_name, format = 'pdf', bbox_inches = 'tight',transparent=True)

   # create dataframe with percentiles of the bootstrapped distribution
   mstar_tmp=grouped_linear['stellar_mass'].quantile([0.05,0.1585,0.5,0.8415,0.95]).unstack(level=1)
   idx=mstar_tmp.columns.to_frame()
   idx.insert(0, '', mstar_str)
   mstar_tmp.columns = pd.MultiIndex.from_frame(idx)
   mstar_tmp.index.rename('mass_range',inplace=True)
   if bs_perc.empty:
      bs_perc=mstar_tmp.copy()
   else:
      bs_perc=bs_perc.copy().join(mstar_tmp)
   perc_colnames=mstar_tmp.columns.get_level_values(1)

   bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row,[0.05,0.1585,0.5,0.8415,0.95]) for row in gals_bs[sfr_str]]), 
                                    index=bs_perc.index, columns=pd.MultiIndex.from_product([[sfr_str],perc_colnames])))
   bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row,[0.05,0.1585,0.5,0.8415,0.95]) for row in gals_bs[lum_str]]), 
                                    index=bs_perc.index, columns=pd.MultiIndex.from_product([[lum_str],perc_colnames])))
   #bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row,[0.05,0.1585,0.5,0.8415,0.95]) for row in gals_bs['unweighted_luminosity']]), 
   #                                  index=bs_perc.index, columns=pd.MultiIndex.from_product([['unweighted_luminosity'],perc_colnames])))
   #bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row['luminosity']/row['SFR'],[0.05,0.1585,0.5,0.8415,0.95]) for i,row in gals_bs.iterrows()]), 
   #                                 index=bs_perc.index, columns=pd.MultiIndex.from_product([['lx_SFR'],perc_colnames])))
   edd_tmp=grouped_linear.edd_ratio.apply(lambda s: pd.DataFrame({
                                             (lambda_str,"mean"): [np.mean(s)],
                                             (lambda_str,"median"): [np.median(s)],
                                             }))


   # remove galaxies with lx < lx_XRB, as of Lehmer+16, from both dfs
   #print(bs_perc[(mstar_str,0.5)])
   #print(bs_perc[(sfr_str,0.5)])
   lx_xrb=10**(29.37+2.03*np.log10(1+z)+bs_perc[(mstar_str,0.5)]-42)+10**(39.28+1.31*np.log10(1+z)+np.log10(bs_perc[(sfr_str,0.5)])-42)
   #print(lx_xrb)
   #α0(1 + z)γ M∗ + β0(1 + z)δSFR, (3)
   #with logα0 = 29.37, logβ0 = 39.28, γ = 2.03, and δ = 1.31.
   for column in bs_perc[lum_str]:
      #bs_perc[lum_str]=bs_perc[lum_str]-lx_xrb
      bs_perc[(lum_str,column)]=bs_perc[(lum_str,column)]-lx_xrb
   #print(bs_perc[lum_str])

   edd_tmp=edd_tmp.droplevel(1)
   bs_perc=bs_perc.join(edd_tmp)
#print(bs_perc.columns)
#print(bs_perc)

# save dataframe to file 
if methods['edd_ratio']=="Schechter":
  bs_perc.to_csv(curr_dir+'/Ros_plots/'+sub_dir+f'bs_perc_z{z}_lambda{lambda_z:.2f}_alpha{alpha_z:.2f}_lambdac{lambda_char:.3f}'+suffix+'.csv')
elif methods['edd_ratio']=="Gaussian":
  bs_perc.to_csv(curr_dir+'/Ros_plots/'+sub_dir+f'bs_perc_z{z}_mean{alpha_z:.2f}_sigma{lambda_z:.2f}_lambdac{lambda_char:.3f}'+suffix+'.csv')

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