
import numpy as np
import weightedstats as ws
from numpy.core.numeric import NaN
import pandas as pd

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

def is_active(duty_cycle):
   tosses=np.random.uniform(size=len(duty_cycle))
   result=np.full_like(duty_cycle, 0.)
   result[tosses <= duty_cycle] = 1.
   return result

def sf_type(size=50,i=0,mbin=0):
   npzfile = np.load('./IDL_data/type_fractions.npz')
   frac_SF=npzfile['frac_SF']
   frac_Q=npzfile['frac_Q']
   frac_SB=npzfile['frac_SB']
   types_list=['SF', 'Q', 'SB']
   data = np.random.choice(  
        a=types_list,  
        size=size,  
        p=[frac_SF[i,mbin], frac_Q[i,mbin], frac_SB[i,mbin]])
   return data

def weighted_quantile(values, sample_weight=None, quantiles=0.5, 
                      values_sorted=False):
   """ Very close to numpy.percentile, but supports weights.
   NOTE: quantiles should be in [0, 1]!
   :param values: numpy.array with data
   :param quantiles: array-like with many quantiles needed
   :param sample_weight: array-like of the same length as `array`
   :param values_sorted: bool, if True, then will avoid sorting of
      initial array
   :param old_style: if True, will correct output to be consistent
      with numpy.percentile.
   :return: numpy.array with computed quantiles.
   """
   values = np.array(values)
   quantiles = np.array(quantiles)
   if sample_weight is None:
      sample_weight = np.ones(len(values))
   sample_weight = np.array(sample_weight)
   assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
      'quantiles should be in [0, 1]'

   if not values_sorted:
      sorter = np.argsort(values)
      values = values[sorter]
      sample_weight = sample_weight[sorter]

   # commented part "centers" the probability on the bin
   weighted_quantiles = np.cumsum(sample_weight) #- 0.5 * sample_weight
   weighted_quantiles /= np.sum(sample_weight)
   return np.interp(quantiles, weighted_quantiles, values)

def my_draw_bs_reps(data,weights=None, size=1,type='median'):
   """
   Generate bootstrap replicates out of `data` using `weightedstats.weighted_median` and `weights`.

   Parameters
   ----------
   data : array_like
      One-dimensional array of data.
   weights : array_like
      One-dimensional array of data of the same size as data.
   size : int, default 1
      Number of bootstrap replicates to generate.

   Returns
   -------
   output : ndarray
      Bootstrap replicates computed from `data` using `weighted_quantile`.

   Notes
   -----
   .. nan values are ignored.
   """
   # Set up output array
   bs_reps = np.empty(size)

   n = len(data)
   if n == 0:
      bs_reps[:]=np.nan
      return bs_reps
   if weights is None:
      weights=np.ones_like(data)

   # Draw replicates
   #print(f'Data points: {n}')
   for i in range(size):
      idx=np.random.choice(np.arange(n,dtype='int'),size=n)
         #bs_reps[i] = weighted_quantile(data[idx], weights[idx])
      if type=='median':
         if isinstance(data,(pd.Series,pd.DataFrame)):
            data_list=data.iloc[idx].tolist()
         else:
            data_list=data[idx].tolist()
         if isinstance(weights,(pd.Series,pd.DataFrame)):
            weights_list=weights.iloc[idx].tolist()
         else:
            weights_list=weights[idx].tolist()
         bs_reps[i] = ws.weighted_median(data_list, weights=weights_list)
      elif type=='mean':
         bs_reps[i] = np.average(data[idx], weights=weights[idx])
         #check=np.sum(data[idx]*weights[idx])/(np.sum(weights[idx]))
         #if check==bs_reps[i]:
         #   print('all good')
         #else:
         #   print('panic!!!')
      else:
         print('Please enter valid type of operation, either mean or median')
         return
   return bs_reps

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
