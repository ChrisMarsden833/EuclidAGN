
from AGNCatalogToolbox import main as agn
from AGNCatalogToolbox import Literature
from colossus.cosmology import cosmology
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dc_stat_think as dcst
from scipy import interpolate

curr_dir=os.getcwd()


################################
# import data from IDL
from scipy.io import readsav
read_data = readsav('vars_EuclidAGN_90.sav',verbose=True)

data={}
for key, val in read_data.items():
    data[key]=np.copy(val)
    data[key][data[key] == 0.] = np.nan
print(data.keys())

################################
# my cosmology
params = {'flat': True, 'H0': 70., 'Om0': 0.3, 'Ob0': 0.0486, 'sigma8':0.8159, 'ns':0.9667}
cosmology.addCosmology('Carraro+20', params)

cosmo = 'Carraro+20'
cosmology = cosmology.setCosmology(cosmo)
volume = 200**3 # Mpc?

################################
# set simulation parameters
z = 1.
reds_dic={0.45:0, 1:1, 1.7:2, 2.7:3}
index=reds_dic.get(z) # needed for IDL data

sub_dir='42_TestSchechter_R&V/'
sub_dir='43_TestSchechter_Shankar/'
sub_dir='44_TestGaussian_Shankar/'
sub_dir='45_TestSchechter_Davis/'
#sub_dir='46_TestSchechter_Sahu/'

methods={'halo_to_stars':'Grylls19', # 'Grylls19' or 'Moster'
    'BH_mass_method':"Davis18", #"Shankar16", "KormendyHo", "Eq4", "Davis18", "Sahu19" and "Reines&Volonteri15"
    'BH_mass_scatter':"Intrinsic", # "Intrinsic" or float
    'duty_cycle':"Schulze", # "Schulze", "Man16", "Geo" or float (0.18)
    'edd_ratio':"Schechter", # "Schechter", "PowerLaw", "Gaussian", "Geo"
    'bol_corr':'Lusso12_modif', # 'Duras20', 'Marconi04', 'Lusso12_modif'
    'SFR':'Carraro20' # 'Tomczak16', "Schreiber15", "Carraro20"
    }

# looping on:
if methods['edd_ratio']=="Schechter":
   # variable_name = r"$\lambda$"
   # par_str= 'lambda'
   variable_name = r"$\alpha$"
   par_str= 'alpha'
elif methods['edd_ratio']=="Gaussian":
   variable_name = r"$\sigma$"
   par_str= 'sigma'
   #variable_name = r"$\mu$"
   #par_str= 'mean'
#parameters = [0.05,0.1,0.2,0.3,0.6]
parameters = [-0.2,-0.1,0.05]

################################
# Edd ratio parameters definition:
if methods['edd_ratio']=='Schechter' and (methods['duty_cycle']=="Schulze" or methods['duty_cycle']=="Geo") and (methods['BH_mass_method']=="Shankar16" or methods['BH_mass_method']=="Davis18" or methods['BH_mass_method']=="Sahu19" or methods['BH_mass_method']=="Reines&Volonteri15"):
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

    alpha_z=-0.5
    lambda_z=-0.1

if z==1 and methods['edd_ratio']=='Schechter' and (methods['duty_cycle']=="Schulze" or methods['duty_cycle']=="Geo") and (methods['BH_mass_method']=="Reines&Volonteri15"):
    # parameters found by testing, see folder 42_TestSchechter_R&V
    alpha_z=1.2
    lambda_z=10

if z==2.7 and methods['edd_ratio']=='Schechter' and (methods['duty_cycle']=="Schulze" or methods['duty_cycle']=="Geo") and (methods['BH_mass_method']=="Reines&Volonteri15"):
    # sucks
    # parameters found by testing, see folder 42_TestSchechter_R&V
    #alpha_z=5
    #lambda_z=8
    alpha_z=-0.5
    lambda_z=-0.2

# Schechter P(lambda), z=1, duty cycle di Schulze + 2015 usando la relazione Mstar-Mbh di K&H +2013 :
if methods['edd_ratio']=='Schechter' and methods['duty_cycle']=="Schulze" and methods['BH_mass_method']=="KormendyHo":
    lambda_z = -0.4
    alpha_z = 0

# Schechter P(lambda), z=1, duty cycle costante e uguale a 0.18 usando la relazione Mstar-Mbh di Shankar + 16:
if z==1 and methods['edd_ratio']=='Schechter' and methods['duty_cycle']==0.18 and methods['BH_mass_method']=="Shankar16":
    lambda_z = -1
    alpha_z = 1.2

if methods['edd_ratio']=='Gaussian':
    sigma_z = 0.05 # sigma
    mu_z = 0.25 # mean edd

#if methods['BH_mass_method']=="Davis18":
#    slope=1.

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
#elif methods['BH_mass_method']=="Davis18":
#    M_inf=10.3
#    M_sup=11.4
#elif methods['BH_mass_method']=="Sahu19":
#    M_inf=10.
#ß    M_sup=12.15
#elif methods['BH_mass_method']=="Reines&Volonteri15":
#    M_inf=10.
print(M_inf,M_sup)

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
    gals=gals[gals['stellar_mass'] <= M_sup]
print(gals.stellar_mass.min(),gals.stellar_mass.max())

# BH
gals['black_hole_mass'] = agn.stellar_mass_to_black_hole_mass(gals.stellar_mass, method = methods['BH_mass_method'], 
                                                                                scatter = methods['BH_mass_scatter'],)#slope=slope,norm=norm

plt.figure()
# XLF Data
print('z=',z)
XLF = Literature.XLFData(z)
mXLF_data = XLF.get_miyaji2015()
plt.plot(mXLF_data.x, mXLF_data.y, 'o', label = "Miyaji")
uXLF_data = XLF.get_ueda14(np.arange(42, 46, 0.1))
plt.plot(uXLF_data.x, uXLF_data.y, ':', label = "Ueda")

for par in parameters:
   if par_str == 'lambda':
      lambda_z=par
   elif par_str == 'alpha':
      alpha_z=par

   # copy df
   gals_tmp= gals.copy()

   # Duty cycles
   gals_tmp['duty_cycle'] = agn.to_duty_cycle(methods['duty_cycle'], gals_tmp.stellar_mass, gals_tmp.black_hole_mass, z, suppress_output=True)

   gals_tmp['luminosity'], XLF_plotting_data, _ = agn.black_hole_mass_to_luminosity(gals_tmp.black_hole_mass, 
                                          gals_tmp.duty_cycle, gals_tmp.stellar_mass, z, methods['edd_ratio'], return_plotting_data=True,
                                          bol_corr=methods['bol_corr'], parameter1=lambda_z, parameter2=alpha_z)
                                          #bol_corr=methods['bol_corr'], parameter1=sigma_z, parameter2=mu_z)
   
   # plot XLF
   #try:
   plt.plot(10**XLF_plotting_data.x, 10**XLF_plotting_data.y, label = r"{} = {}".format(variable_name, par))
   #except:
   #   print('passing XLF plotting')
   #   pass

   """
   # do squared distance from Miyaji
   XLF_plot_int=interpolate.interp1d(10**XLF_plotting_data.x, 10**XLF_plotting_data.y)
   good_ones=np.logical_and((mXLF_data.x>=10**XLF_plotting_data.x[0]), (mXLF_data.x<=10**XLF_plotting_data.x[-1]))
   mXLF_x=mXLF_data.x[good_ones]
   XLF_sim=XLF_plot_int(mXLF_x)
   S=np.sum((XLF_sim-mXLF_data.y[good_ones])**2)
   print(f'squares sum for {par_str}={par}:\t{S}')
   """

   #gals_tmp['nh'] = agn.luminosity_to_nh(gals_tmp.luminosity, z)
   #gals_tmp['agn_type'] = agn.nh_to_type(gals_tmp.nh)

   gals_tmp['SFR'] = agn.SFR(z,gals_tmp.stellar_mass,methods['SFR'])
   gals_tmp['lx/SFR'] = (gals_tmp.luminosity-42)-gals_tmp.SFR


# finish plot
plt.xlabel(r'$L_x\;[erg\;s^{-1}]$')
plt.ylabel(r'$d\phi /d(log\;L_x)\;[Mpc^{-3}]$')
plt.loglog()
plt.legend()
if methods['edd_ratio']=="Schechter":
   if par_str == 'lambda':
      title_str=r'XLF, U={}, $\alpha$={}, z={}'.format(methods['duty_cycle'],alpha_z,z)
      file_name=curr_dir+'/Ros_plots/'+sub_dir+f'XLF_z{z}_alpha{alpha_z}'+par_str+'.pdf'
   elif par_str == 'alpha':
      title_str=r'XLF, U={}, $\lambda$={}, z={}'.format(methods['duty_cycle'],lambda_z,z)
      file_name=curr_dir+'/Ros_plots/'+sub_dir+f'XLF_z{z}_lambda{lambda_z}'+par_str+'.pdf'
elif methods['edd_ratio']=="Gaussian":
   if par_str == 'sigma':
      title_str=r'XLF, U={}, $\mu$={}, z={}'.format(methods['duty_cycle'],mu_z,z)
      file_name=curr_dir+'/Ros_plots/'+sub_dir+f'XLF_z{z}_mean{mu_z}'+par_str+'.pdf'
   elif par_str == 'mean':
      title_str=r'XLF, U={}, $\sigma$={}, z={}'.format(methods['duty_cycle'],sigma_z,z)
      file_name=curr_dir+'/Ros_plots/'+sub_dir+f'XLF_z{z}_sigma{sigma_z}'+par_str+'.pdf'
plt.title(title_str)
plt.savefig(file_name, format = 'pdf', bbox_inches = 'tight',transparent=True)
#plt.show()
#print(gals.describe())

################################
# grouping in mass bins - log units
grouped_gals = gals_tmp[['stellar_mass','luminosity','SFR','lx/SFR']].groupby(pd.cut(gals_tmp.stellar_mass, np.append(np.arange(5, 11.5, 0.5),12.))).quantile([0.05,0.1585,0.5,0.8415,0.95]).unstack(level=1)

# converting to linear units
gals_lin=pd.DataFrame()
gals_lin['stellar_mass'] = gals_tmp['stellar_mass']
gals_lin['luminosity']= 10**(gals_tmp.luminosity-42)
gals_lin[['SFR','lx/SFR']]=10**gals_tmp[['SFR','lx/SFR']]

# grouping linear table
grouped_lin = gals_lin[['stellar_mass','luminosity','SFR','lx/SFR']].groupby(pd.cut(gals_tmp.stellar_mass, np.append(np.arange(5, 11.5, 0.5),12.))).quantile([0.05,0.1585,0.5,0.8415,0.95]).unstack(level=1)
# limit to logM>9
ggals_lin=grouped_lin[grouped_lin['stellar_mass',0.5] > 9]
grouped_lin.index.rename('mass_range',inplace=True)

################################
## Bootstrapping ##
################################
func=np.median
M_min=np.max([9,M_inf])

# create dataframe for bootstrapping
gals_highM=gals_lin.copy()[gals_lin.stellar_mass > M_min]
grouped_linear = gals_highM[['stellar_mass','luminosity','SFR','lx/SFR']].groupby(pd.cut(gals_highM.stellar_mass, np.append(np.arange(M_min, 11.5, 0.5),12.)))#.quantile([0.05,0.1585,0.5,0.8415,0.95]).unstack(level=1)

# create dataframe of bootstraped linear varibles
gals_bs=pd.DataFrame()
gals_bs['SFR'] = grouped_linear.SFR.apply(lambda x: dcst.draw_bs_reps(x, func, size=500))
gals_bs['luminosity'] = grouped_linear.luminosity.apply(lambda x: dcst.draw_bs_reps(x, func, size=500))
gals_bs.head()

# create dataframe with percentiles of the bootstrapped distribution
bs_perc=ggals_lin['stellar_mass']
old_idx = bs_perc.columns.to_frame()
perc_colnames=bs_perc.columns
old_idx.insert(0, '', 'stellar_mass')
bs_perc.columns = pd.MultiIndex.from_frame(old_idx)

bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row,[0.05,0.1585,0.5,0.8415,0.95]) for row in gals_bs['SFR']]), 
                                  index=bs_perc.index, columns=pd.MultiIndex.from_product([['SFR'],perc_colnames])))
bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row,[0.05,0.1585,0.5,0.8415,0.95]) for row in gals_bs['luminosity']]), 
                                  index=bs_perc.index, columns=pd.MultiIndex.from_product([['luminosity'],perc_colnames])))
bs_perc=bs_perc.join(pd.DataFrame(np.array([np.quantile(row['luminosity']/row['SFR'],[0.05,0.1585,0.5,0.8415,0.95]) 
                                             for i,row in gals_bs.iterrows()]), 
                                  index=bs_perc.index, columns=pd.MultiIndex.from_product([['lx_SFR'],perc_colnames])))
#print(bs_perc)

# save dataframe to file and add to dictionary for use
bs_perc.to_csv(curr_dir+'/Ros_plots/'+sub_dir+f'bs_perc_z{z}.csv')
df_dict[par_str+f'{par}']=bs_perc

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
plt.savefig(curr_dir+'/Ros_plots/'+sub_dir+f'SFvsLX_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True) 