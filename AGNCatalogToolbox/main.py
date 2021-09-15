# Common
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy import interpolate
from multiprocessing import Process, Value, Array, Pool
import multiprocessing
import time
import pandas as pd
from scipy.integrate import quad

# Specialized
from colossus.lss import mass_function
from colossus.cosmology import cosmology
from Corrfunc.theory import wp

# Local
from AGNCatalogToolbox import Utillity as utl
from AGNCatalogToolbox import ImageGenerator as img


def generate_semi_analytic_halo_catalogue(catalogue_volume, mass_params=(12, 16, 0.1), z=0, h=0.7):
    """ Function to generate the semi analytic halo catalogue (without coordinates) for galaxy testing
    :param catalogue_volume: float, cosmological volume within which to generate the catalog [Mpc^3] **NOT [Mpc^3 h^-3]**
    :param mass_params: tuple, (mass low, mass high, spacing) [log10 M_sun].
    :param z: float, redshift [dimensionless]
    :param h: float, reduced hubble constant [dimensionless]
    :return array, of halo masses [log10 M_sun]
    """

    # We need the volume in h^-1 to interface properly with colossus
    axis = catalogue_volume**(1/3)  # Mpc
    axish = axis*h  # Mpc/h
    volume_h = axish**3  # Mpc^3 h^-3

    # Get the bin width and generate the bins. This also needs to be in h^-1
    bin_width = mass_params[2]
    mass_range = 10 ** np.arange(mass_params[0], mass_params[1], mass_params[2]) + np.log10(h)  # h^-1

    # Generate the mass function itself - this is from the colossus toolbox
    local_mass_function = mass_function.massFunction(mass_range, z, mdef='200m', model='tinker08', q_out='dndlnM') \
        * np.log(10) / h  # dn/dlog10M

    # We determine the Cumulative HMF starting from the high mass end, multiplied by the bin width.
    # This effectively gives the cumulative probability of a halo existing.
    cumulative_mass_function = np.flip(np.cumsum(np.flip(local_mass_function, 0)), 0) * bin_width

    # Multiply by volume
    cumulative_mass_function = cumulative_mass_function * volume_h

    # Get the maximum cumulative number.
    max_number = np.floor(cumulative_mass_function[0])
    range_numbers = np.arange(max_number)

    # Update interpolator
    interpolator = interpolate.interp1d(cumulative_mass_function, mass_range)
    mass_catalog = interpolator(range_numbers[range_numbers >= np.amin(cumulative_mass_function)])

    mass_catalog = np.log10(mass_catalog/h)
    return mass_catalog

def load_halo_catalog(path="auto", z=0.0, h=0.7, filename_prefix="MD_", path_big_data="./BigData/", user="Chris"):
    """ Function to load in the catalog_data from the multi-dark halo catalogue
    This catalog_data should exist as .npy files in the Directory/BigData. Within this folder there should be a script
    to pull these out of the cosmosim database. Note that this expects a .npy file in the with columns representing x,
    y, z, scale factor at accretion, mass at accretion, virial mass, main and upid.
    :param path: string, direct path and name of the file. If set to "auto" (the default), this will find the file for
        you within the path_big_data folder.
    :param z: float, redshift [dimensionless]. Only used when path == "auto"
    :param h: float, reduced hubble constant [dimensionless]
    :param filename_prefix: string, component of the filename excluding the redshift - the closest z will be found.
        Only used when path == "auto". Default is "MD_", expecting files of the form "MD_0.0.npy".
    :param path_big_data: string, path to the location of the data. Only used when path == "auto"
    :param user: string. This is an adjustable string purely to trigger an if condition that allows Viola to use her
        catalogs quickly. Defaults to "Chris".
    :return effective_halo_mass (array) [log10 M_sun], effective_z (array) [dimensionless], virial_mass (array) [log10
        M_sun], up_id (array) [integer], x, y, z (arrays) [Mpc].
    """
    # Path finding logic
    if path == "auto":
        print("Searching for file...")
        # Search for the file with the closest z and read it in.
        catalog_file, catalog_z = utl.GetCorrectFile(filename_prefix, z, path_big_data, True)
        print("Found file:", catalog_file)
        catalog_data = np.load(path_big_data + catalog_file)
        print("dtypes found: ", catalog_data.dtype)
    else:
        print("Attempting to load file:" + path)
        catalog_data = np.load(path)

    # This is just to help out Viola, her files are slightly different.
    if user == "Viola":
        data_x = catalog_data['x']/h
        data_y = catalog_data['y']/h
        data_z = catalog_data['z']/h
        main_id = catalog_data['rockstarId']
        up_id = catalog_data['upId']
        virial_mass = catalog_data['Mvir']/h
        mass_at_accretion = catalog_data['irst_Acc_MvirF']/h
        accretion_scale = catalog_data['irst_Acc_Scale']
    else:  # Everyone else should change their dtypes to this naming convention
        data_x = catalog_data['x']/h
        data_y = catalog_data['y']/h
        data_z = catalog_data['z']/h
        main_id = catalog_data["id"]
        up_id = catalog_data["upid"]
        virial_mass = catalog_data["mvir"]/h
        mass_at_accretion = catalog_data["First_Acc_Mvir"]/h
        accretion_scale = catalog_data["First_Acc_Scale"]

    del catalog_data  # Save on memory. This is actually meaningful in some cases.

    print("Starting the sorting of data")

    # Reserved memory
    virial_mass_parent = np.zeros_like(virial_mass)
    idh = np.zeros_like(virial_mass)
    effective_z = np.zeros_like(accretion_scale)

    print('    Sorting halos by upId')
    sorted_indexes = up_id.argsort()  # + 1 Array, rest are reused
    data_x = data_x[sorted_indexes]
    data_y = data_y[sorted_indexes]
    data_z = data_z[sorted_indexes]
    # We also do this with the local data
    main_id = main_id[sorted_indexes]
    up_id = up_id[sorted_indexes]
    virial_mass = virial_mass[sorted_indexes]
    mass_at_accretion = mass_at_accretion[sorted_indexes]
    accretion_scale = accretion_scale[sorted_indexes]
    up_id_0 = np.searchsorted(up_id, 0)  # Position where zero should sit on the now sorted up_id.
    # All arrays are should now sorted by up_id.

    print('    Copying all {} elements with up_id = -1'.format(str(up_id_0)))
    virial_mass_parent[:up_id_0] = virial_mass[:up_id_0]  # MVir of centrals, or where up_id  = -1.

    print('    Sorting remaining list by main id')
    up_id_cut = up_id[up_id_0:]  # Up_id's that are not -1, or the satellites, value pointing to their progenitor.
    id_cut = main_id[:up_id_0]  # ids of centrals
    virial_mass_cut = virial_mass[:up_id_0]  # masses of centrals
    sorted_indexes = id_cut.argsort()  # get indexes to centrals by id.
    id_cut = id_cut[sorted_indexes]  # actually sort centrals by id.
    virial_mass_cut = virial_mass_cut[sorted_indexes]  # sort virial masses the same way.

    print('    Copying remaining', str(len(up_id) - up_id_0), 'elements')
    sorted_indexes = np.searchsorted(id_cut, up_id_cut)  # indexes of where satellite id's point to centrals
    virial_mass_parent[up_id_0:] = virial_mass_cut[sorted_indexes]  # Sort parents by this, and assign to satellites
    # This gives us the virial mass of the parent or itself if it is a parent. But do we actually need this?
    idh[up_id_0:] = 1

    halo_mass = virial_mass
    halo_mass[idh > 0] = mass_at_accretion[idh > 0]
    effective_z[idh > 0] = 1 / accretion_scale[idh > 0] - 1
    effective_z[idh < 1] = catalog_z

    effective_halo_mass = np.log10(halo_mass)
    # Effective halo mass is the virial mass of centrals and the mass at accretion of satallites

    return effective_halo_mass, effective_z, np.log10(virial_mass), up_id, data_x, data_y, data_z

def halo_mass_function(halo_masses, bins, volume, h=0.7, compare=False, z=0):
    """ Function to generate the HMF from an array of halo masses. Note that input units are [log10 M_sun] and
    returned units are [log10 M_sun h^-1 ]. This is because hmf's are normally units of h^-1
    :param halo_masses: (numpy) array of halo masses [log10 M_sun].
    :param bins: (numpy) array of bins for halo masses [log10 M_sun].
    :param volume: float, the cosmological volume [MPc^3].
    :param compare: bool, if a colossus HMF should be returned as a 2nd argument (for comparison).
    :param z: float, redshift of interest, only used if compare == True.
    :param h: float, reduced hubble constant, only used if compare == True.
    :return: 2 (4) arguments if compare is False (True)
        hmf: the halo mass function [log10 dN/dlog10_M_halo h^-3 Mpc^-3].
    :
    """
    if halo_masses.any() > 100:
        raise ValueError("Check units, Halo masses were supplied with masses > 100 log10 M_sun. ")
    elif bins.any() > 100:
        raise ValueError("Check units, Bins were supplied with masses > 100 log10 M_sun")
    if not utl.evenly_spaced(bins):
        raise ValueError("The spacing of the bins is not even")

    axis = volume ** (1 / 3)  # Mpc
    axish = axis * h  # Mpc/h
    volume_h = axish ** 3  # Mpc^3 h^-3

    bins_h = bins + np.log10(h)  # log10 h^-1
    halo_masses_h = halo_masses + np.log10(h)  # log10  h^-1

    hist = np.histogram(halo_masses_h, bins=bins_h)[0]
    hmf = (hist / volume_h) / abs(bins[1] - bins[0])
    flag = hmf != 0
    hmf[flag] = np.log10(hmf[flag])
    hmf[hmf == 0] = float('NaN')

    if compare:
        colossus_mf = np.log10(mass_function.massFunction(10**bins_h, z, mdef='200m', model='tinker08',
                                                         q_out='dndlnM') * np.log(10)/h)  # dn/dlog10M
        return hmf, bins_h[:-1], colossus_mf, bins_h
    return hmf, bins_h[:-1]

def halo_mass_to_stellar_mass(halo_mass, z=0, formula="Grylls19", scatter=0.11):
    """Function to generate stellar masses from halo masses.

    This is based on Grylls 2019, but also has the option to use the
    parameters from Moster. This is a simplified version of Pip's
    DarkMatterToStellarMass() function.

    :param halo_mass: (numpy) array, of halo masses [log10 M_sun].
    :param z: float, the value of redshift [dimensionless]. Default is zero.
    :param formula: string, the method to use. Options currently include "Grylls19" and "Moster".
    :param scatter: float or bool, scatter magnitude [log10 M_sun], or if set to False will evaluate with no scatter.
    :return (numpy) array, of stellar masses [log10] .
    """

    # If conditions to set the correct parameters.
    if formula == "Grylls19":
        z_parameter = np.divide(z - 0.1, z + 1)
        m_10, shm_norm_10, beta10, gamma10 = 11.95, 0.032, 1.61, 0.54
        m_11, shm_norm_11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
    elif formula == "Moster":
        z_parameter = np.divide(z, z + 1)
        m_10, shm_norm_10, beta10, gamma10 = 11.590, 0.0351, 1.376, 0.608
        m_11, shm_norm_11, beta11, gamma11 = 1.195, -0.0247, -0.826, 0.329
    else:
        raise ValueError("Unrecognised formula parameter, got: {}".format(formula))

    # Create full parameters
    m = m_10 + m_11 * z_parameter
    n = shm_norm_10 + shm_norm_11 * z_parameter
    b = beta10 + beta11 * z_parameter
    g = gamma10 + gamma11 * z_parameter
    # Full formula
    internal_stellar_mass = np.log10(np.power(10, halo_mass) *
                                     (2 * n * np.power((np.power(np.power(10, halo_mass - m), -b)
                                                        + np.power(np.power(10, halo_mass - m), g)), -1)))
    # Add scatter, if requested.
    if not scatter == False:
        print("Scatter is a thing, valued at {}".format(scatter))
        internal_stellar_mass += np.random.normal(scale=scatter, size=np.shape(internal_stellar_mass))
        # add measure error (for comparison with data)
        internal_stellar_mass += np.random.normal(scale=.15, size=np.shape(internal_stellar_mass))
    return internal_stellar_mass

def stellar_mass_to_black_hole_mass(stellar_mass, method="Shankar16", scatter="Intrinsic",slope=1.05,norm=7.45):
    """ Function to assign black hole mass from the stellar mass.
    :param stellar_mass: array, of stellar masses [log10 M_sun]
    :param method: string, specifying the method to be used, current options are "Shankar16",  "KormondyHo" and "Eq4".
    :param scatter: string or float, string should be "Intrinsic", float value specifies the (fixed) scatter magnitude
    :slope: float, used to test Reines&Volonteri15 relation
    :norm: float, used to test Reines&Volonteri15 relation
    :return: array, of the black hole masses [log10 M_sun].
    """

    # Main formula
    if method == "Shankar16":
        log_black_hole_mass = 7.574 + 1.946 * (stellar_mass - 11) - 0.306 * (stellar_mass - 11)**2. \
                              - 0.011 * (stellar_mass - 11)**3
    elif method == "KormondyHo":
        log_black_hole_mass = 8.54 + 1.18 * (stellar_mass - 11)
    elif method == 'Eq4':
        log_black_hole_mass = 8.35 + 1.31 * (stellar_mass - 11)
    elif method == "Davis18":
       # Davis+18, eq. 3
       log_black_hole_mass = 7.25 + 3.05 * (stellar_mass - 10.8) # original equation
    elif method == "Sahu19":
       # Sahu+19, Eq. 11, fig. 11 
       log_black_hole_mass = 8.02 + 1.65 * (stellar_mass - 10.7)
    elif method == "Reines&Volonteri15":
       # Eq. 4,5 and Fig 8 
       # log_black_hole_mass = 7.45 + 1.05 * (stellar_mass - 11.) # original equation
       log_black_hole_mass = norm + slope * (stellar_mass - 11.)
    else:
        raise ValueError("Unknown method {}".format(scatter))

    # Scatter
    if scatter == "Intrinsic" or scatter == "intrinsic":
        if method == "Shankar16":
            # The intrinsic formula for scatter given in FS's relation
            scatter = (0.32 - 0.1 * (stellar_mass - 12.)) * np.random.normal(0., 1., len(stellar_mass))
        elif method == "Reines&Volonteri15":
            print('Scatter R&V')
            scatter = np.random.normal(0, 0.55, len(stellar_mass)) # rms deviation, pag8
        elif method == "Sahu19":
            print('Scatter Sahu19')
            scatter = np.random.normal(0, 0.58, len(stellar_mass)) # delta_rms, pag13, table 5
        elif method == "Davis18":
            print('Scatter Davis18')
            scatter = np.random.normal(0, 0.79, len(stellar_mass)) #delta_rms, pag9
        else:
            print('Scatter=0.5')
            scatter = np.random.normal(0, 0.5, len(stellar_mass))
    elif isinstance(type(scatter), float):
        scatter = np.random.normal(0., scatter, len(stellar_mass))
    elif isinstance(type(scatter), list) or isinstance(type(scatter), np.ndarray):
        scatter = np.random.normal(0, 1, len(stellar_mass)) * scatter
    elif scatter == False or scatter == None:  # Not sketchy, as we actually want to compare this with a boolean,
        # ...despite what my IDE thinks.
        scatter = np.zeros_like(stellar_mass)
    else:
        raise ValueError("Unknown Scatter argument {}".format(scatter))

    log_black_hole_mass += scatter
    return log_black_hole_mass

def to_duty_cycle(method, stellar_mass, black_hole_mass, z=0, data_path="./Data/DutyCycles/"):
    """ Function to assign duty cycle.
    :param method: string/float. If string, should be a method (currently "Man16" or "Schulze").
        If float will be the value of a constant duty cycle, e.g. 0.1 [dimensionless]
    :param stellar_mass: array, the stellar masses, [log10 M_sun].
    :param black_hole_mass: array, the black hole masses [log10 M_sun].
    :param z: float, redshift [dimensionless]
    :param data_path: string, path to the directory where the data is stored.
    :return: array, the duty cycle [dimensionless]
    """

    plt.figure()

    method_type = type(method)
    if method_type is float or method_type is int:
        duty_cycle = np.ones_like(stellar_mass) * method
    elif isinstance(method, str):
        if method == "Man16":
            if z > 0.1:
                print("Warning - Mann's duty cycle is not set up for redshifts other than zero")
            mann_path = data_path + "Mann.csv"
            df = pd.read_csv(mann_path, header=None)
            mann_stellar_mass = df[0].values
            mann_duty_cycle = df[1].values
            get_u = interpolate.interp1d(mann_stellar_mass, mann_duty_cycle, bounds_error=False,
                                            fill_value=(mann_duty_cycle[0], mann_duty_cycle[-1]))
            duty_cycle = get_u(stellar_mass)
            plt.plot(stellar_mass, duty_cycle, '.', label = "Mocks",alpha=0.2)
            plt.plot(mann_stellar_mass, mann_duty_cycle, '-', label = "Model")
            plt.xlabel(r'$M_*\;[M_\odot]$')

        elif method == "Schulze":
            # Find the nearest file to the redshift we want
            schulze_path = data_path + utl.GetCorrectFile("Schulze", z, data_path)
            df = pd.read_csv(schulze_path, header=None)
            df.sort_values(by=[0], inplace=True)  # sort by MBH
            schulze_black_hole_mass = df[0].values
            schulze_duty_cycle = df[1].values
            get_u = interpolate.interp1d(schulze_black_hole_mass, schulze_duty_cycle, bounds_error=False,
                                            fill_value=(schulze_duty_cycle[0], schulze_duty_cycle[-1]))
            if (get_u(np.max(black_hole_mass)) > np.min(schulze_duty_cycle)) or (get_u(np.min(black_hole_mass)) < np.max(schulze_duty_cycle)):
               print('WARNING: Something is wrong with the duty cycle')
            #print(f'U({np.max(black_hole_mass)})={get_u(np.max(black_hole_mass))}, U({np.min(black_hole_mass)})={get_u(np.min(black_hole_mass))}')
            duty_cycle = 10 ** get_u(black_hole_mass)
            plt.plot(black_hole_mass, duty_cycle, '.', label = "Mocks",alpha=0.2)
            plt.plot(schulze_black_hole_mass, 10**schulze_duty_cycle, '-', label = "Model")
            plt.xlabel(r'$M_{BH}\;[M_\odot]$')
            
        elif method == "Geo":
            geo_stellar_mass, geo_duty_cycle = utl.ReadSimpleFile("Geo17DC", z, data_path)
            get_u = interpolate.interp1d(geo_stellar_mass, geo_duty_cycle, bounds_error=False,
                                            fill_value=(geo_duty_cycle[0], geo_duty_cycle[-1]))

            duty_cycle = 10 ** get_u(stellar_mass)
            plt.plot(stellar_mass, duty_cycle, '.', label = "Mocks",alpha=0.2)
            plt.plot(geo_stellar_mass, 10**geo_duty_cycle, '-', label = "Model")
            plt.xlabel(r'$M_*\;[M_\odot]$')
        else:
            assert False, "Unknown Duty Cycle Type {}".format(method)
    else:
        assert False, "No duty cycle type specified"

    plt.legend()
    plt.ylabel('Duty cycle')
    plt.yscale('log')
    plt.title(f'{method}, z={z}')
    plt.savefig(f'Duty_cycles_{method}_z{z}.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True)
  
    assert len(duty_cycle[(duty_cycle < 0) * (duty_cycle > 1)]) == 0, \
        "{} Duty Cycle elements outside of the range 0-1 exist. This is a probability, so this is not valid. Values: {}"\
        .format(len(duty_cycle[(duty_cycle < 0) * (duty_cycle > 1)]), duty_cycle[(duty_cycle < 0) * (duty_cycle > 1)])

    return duty_cycle

def edd_schechter_function(edd, method="Schechter", arg1=-1, arg2=-0.65, arg3=1, redshift_evolution=False, z=0, data_path="./Data/"):
    #gammaE = arg2
    #A = 10. ** (-1.41)
    #prob = ((edd / (10. ** arg1)) ** arg2)

    if redshift_evolution:
        gammaz = 3.47
        z0 = 0.6
        prob *= ((1. + z) / (1. + z0)) ** gammaz

    if method == "Schechter":
        def Schechter(x,llambda,alpha):
            return 10.**(-alpha*(x-llambda))*np.exp(-(10.**(x-llambda)))*10.**x*np.log(10.)

        norm_fac=quad(Schechter, np.min(edd), np.max(edd), args=(arg1,arg2))
        #norm_fac1=np.sum(Schechter(edd,arg1,arg2)*(edd[1]-edd[0]))

        """
        # plot schechter and cumulative
        plt.figure()
        #plt.yscale('log')
        plt.plot(edd,Schechter(edd,arg1,arg2),label='Schechter')
        plt.plot(edd,Schechter(edd,arg1,arg2)/norm_fac[0],label='Norm Schechter mio')
        #plt.plot(edd,Schechter(edd,arg1,arg2)/norm_fac1,label='Norm Schechter Viola')
        plt.plot(edd,np.cumsum(Schechter(edd,arg1,arg2)/norm_fac[0]*(edd[1]-edd[0])),label='Cumulative')
        plt.legend()
        plt.savefig('./Ros_plots/Schechter_distrib.pdf', format = 'pdf', bbox_inches = 'tight',transparent=True)
        """

        return Schechter(edd,arg1,arg2)/norm_fac[0]
    elif method == "PowerLaw":
        def powerlaw(x,gamma1,gamma2,lbreak):
           return ((10**x/lbreak)**gamma1+(10**x/lbreak)**gamma2)**(-1)

        norm_fac=quad(powerlaw, np.min(edd), np.max(edd), args=(arg1,arg2,arg3))
        return powerlaw(edd,arg1,arg2)/norm_fac[0]

    elif method == "Gaussian":
        def Gaussian(x,sigma,mean):
           return np.exp(-(x - mean) ** 2. / (2.*sigma ** 2.))

        norm_fac=quad(Gaussian, np.min(edd), np.max(edd), args=(arg1,arg2))

        # plot Gaussian and cumulative
        #plt.figure()
        #plt.plot(edd,Gaussian(edd,arg1,arg2),label='Gaussian')
        #plt.plot(edd,np.cumsum(Gaussian(edd,arg1,arg2)/norm_fac[0]*(edd[1]-edd[0])),label='Cumulative')
        #plt.plot(edd,Gaussian(edd,arg1,arg2)/norm_fac[0],label='Norm Gaussian')
        #plt.legend()
        #plt.show()

        return Gaussian(edd,arg1,arg2)/norm_fac[0]
        #return np.exp(-(edd - arg2) ** 2. / (2.*arg1 ** 2.))
    elif method == "Geo":
        geo_ed, geo_phi_top, geo_phi_bottom, z_new = utl.ReadSimpleFile("Geo17", z, data_path, cols=3, retz=True)
        mean_phi = (geo_phi_top + geo_phi_bottom)/2
        get_phi = interpolate.interp1d(geo_ed, mean_phi, bounds_error=False, fill_value=(mean_phi[0], mean_phi[-1]))
        return get_phi(edd)
    else:
        assert False, "Type is unknown"

def black_hole_mass_to_luminosity(black_hole_mass, z=0, method="Schechter",
                                  bol_corr='Duras20',
                                  redshift_evolution=False,
                                  parameter1=-1, parameter2=-0.65,parameter3=1,lambda_min=-4):
    """ Function to assign the eddington ratios and X-ray luminosity (2-10KeV) from black hole mass.

    :param black_hole_mass: array, the black hole mass (log10)
    :param z: float, redshift
    :param method: string, the function to pull the Eddington Ratios from. Current options are "Schechter", "PowerLaw"
        or "Gaussian".
    :param redshift_evolution: bool, if set to true will introduce a factor representing the z-evolution.
    :param parameter1: the first parameter for the method. For Schechter it is the knee, for PowerLaw it is not used,
    and for the Gaussian it is sigma.
    :param parameter2: the second parameter for the method. For Schechter it is alpha, for PowerLaw it is not used, for
    Gaussian it is b.
    :return luminosity: (numpy) array, the x-ray luminosity [erg s^-1]
            eddington radio: (numpy) array, the eddington ratio [dimensionless]
            lambda_char: characteristic lambda
    """
    
    l_edd = 38.1072 + black_hole_mass # log(L_Edd) [erg/s]
    
    #plt.figure()
    #plt.hist(l_edd,density=True)
    #plt.ylabel('Mbh+ Ledd')
    #plt.show()

    binning=0.0001
    edd_bin = np.arange(lambda_min, 1, binning)
    #prob_schechter_function = edd_schechter_function(10 ** edd_bin, method=method, arg1=parameter1, arg2=parameter2,
    prob_schechter_function = edd_schechter_function(edd_bin, method=method, arg1=parameter1, arg2=parameter2, arg3=parameter3,
                                                   redshift_evolution=redshift_evolution, z=z)
    #p = prob_schechter_function# * (10**0.0001)
    #r_prob = p[::-1]
    #prob_cum = np.cumsum(r_prob)
    #r_prob_cum = prob_cum[::-1]
    #y = r_prob_cum / r_prob_cum[0]
    #y = y[::-1]
    #edd_bin = edd_bin[::-1]
    y=np.cumsum(prob_schechter_function*binning) # Pn(x)*dlog(x)
    if (y[-1] >1.01) or (y[-1]<0.99):
      print(f'ATTENTION!!! Normalization went wrong, cumulative goes up to {y[-1]} instead of 1')
   
    # characteristic lambda
    lambda_char=np.cumsum(prob_schechter_function*(10**edd_bin)*binning) # Pn(x)*x*dlog(x)  # x=lambda (lin scale)
    if method=="Schechter":
      #lin_bin=np.diff(10**edd_bin, append=0.0023021)#, prepend=2.3025e-08
      #lambda_char=np.cumsum(prob_schechter_function*(10**edd_bin)*lin_bin) # Pn(x)*x*dx  # x=lambda (lin scale)
      #print(f'characteristic lambda (lin) for this eddington distribution: {lambda_char[-1]}')
      #lambda_char=np.cumsum(prob_schechter_function*(10**edd_bin)*binning) # Pn(x)*x*dlog(x)  # x=lambda (lin scale)
      print(f'characteristic log(lambda)={np.log10(lambda_char[-1]):.23} for alpha={parameter2:2f}, lambda={parameter1:2f}')
      #lambda_char=np.cumsum(prob_schechter_function*edd_bin*binning) # Pn(x)*log(x)*dlog(x)
      #print(f'characteristic log(lambda) for alpha={parameter2:2f}, lambda={parameter1:2f}: {lambda_char[-1]:.2f}')
    else: # Gaussian
      #lambda_char=np.cumsum(prob_schechter_function*(10**edd_bin)*binning) # Pn(x)*x*dlog(x)
      print(f'characteristic log(lambda)={np.log10(lambda_char[-1]):.3f} for mean={parameter2:2f}, sigma={parameter1:2f}')
      #lambda_char=np.cumsum(prob_schechter_function*edd_bin*binning) # Pn(x)*log(x)*dlog(x)
      #print(f'characteristic log(lambda) for mean={parameter2:2f}, sigma={parameter1:2f}: {lambda_char[-1]:.2f}')

    a = np.random.random(len(black_hole_mass))
    y2edd_bin = interpolate.interp1d(y, edd_bin, bounds_error=False, fill_value=(edd_bin[0], edd_bin[-1]))
    lg_edd = y2edd_bin(a)  # lgedd = np.interp(a, y, edd_bin)  # , right=-99)
    l_bol = lg_edd + l_edd
    #np.save('l_bol.npy',l_bol)
    #print(f'lum_min={np.min(l_bol)}, lum_max={np.max(l_bol)}')

    """
    plt.figure()
    plt.hist(lg_edd,density=True)
    plt.ylabel('labda distrib')
    plt.show()

    plt.figure()
    plt.hist(l_bol,density=True)
    plt.ylabel('l_bol')
    plt.show()
    """

    if bol_corr == 'Marconi04':
       #eq 21
       lg_l_bol = l_bol - 33.49 #convert to L_sun
       lg_lum = lg_l_bol - 1.54 - (0.24 * (lg_l_bol - 12.)) - \
                (0.012 * ((lg_l_bol - 12.) ** 2.)) + (0.0015 * ((lg_l_bol - 12.) ** 3.))
       luminosity = lg_lum + 33.49 #convert to erg/s

    elif bol_corr == 'Lusso12_modif':
      incr=0.01
      #Fits from table2, type2, Spectro+photo,488
      #range from fig 9 Lbol=[9.8-12.2]
      pars_t2=[0.23, 0.05, 0.001, 1.256]
      Lbol2=np.arange(start=9.8,stop=12.2, step=incr)
      bol_corr2=lusso(Lbol2-12,*pars_t2)
      Lbol2_erg = Lbol2 + 33.585 #from Lsun to erg/s

      #Fits from table2, type1, Spectro+photo,373
      # range from fig 9 Lbol=[10.8-13.2]
      pars_t1 = [0.288, 0.111, -0.007, 1.308]
      L_max = 13.55 # where bol_corr1=100
      Lbol1 = np.arange(start=10.8,stop=L_max, step=incr)
      bol_corr1 = lusso(Lbol1-12,*pars_t1)
      Lbol1_erg = Lbol1 + 33.585 #from Lsun to erg/s

      #Combine equations, eq 1 from Lusso 2012
      start = np.min(Lbol1)
      ending = np.max(Lbol2)
      LX = np.arange(start=start,stop=ending, step=incr)
      bol_corr12 = lusso(LX-12,*pars_t1)
      bol_corr22 = lusso(LX-12,*pars_t2)
      bol_tot = bol_corr12*(LX-start)/(ending-start)+bol_corr22*(ending-LX)/(ending-start)
      Lbol_tot = LX + 33.585 #from Lsun to erg/s

      # Low luminosity (She et al. 2017)
      lx_final = np.arange(start=20.,stop=np.min(Lbol2_erg), step=incr)
      low_lum = np.log10(16)*np.ones(len(lx_final))

      # Compose to make one array:
      # low lum + AGN2
      a = tuple([Lbol2_erg < np.min(Lbol1_erg)])
      lx_final = np.concatenate((lx_final,Lbol2_erg[a]))
      corr_final = np.concatenate((low_lum,bol_corr2[a]))
      # + intermidiate area
      lx_final = np.concatenate((lx_final,Lbol_tot))
      corr_final = np.concatenate((corr_final,bol_tot))
      # + AGN1 up to 100
      a = tuple([Lbol1_erg > np.max(Lbol2_erg)])
      lx_final = np.concatenate((lx_final,Lbol1_erg[a]))
      corr_final = np.concatenate((corr_final,bol_corr1[a]))
      # + flat area
      lx_high = np.arange(start=max(Lbol1_erg),stop=53.,step=incr)
      corr_high = 2*np.ones(len(lx_high))
      lbol_final = np.concatenate((lx_final,lx_high))
      corr_final = np.concatenate((corr_final,corr_high))

      #plt.plot(Lbol2_erg,10**bol_corr2) #type2
      #plt.plot(Lbol1_erg,10**bol_corr1) #type1
      #plt.plot(Lbol_tot,10**bol_tot) #combination of previous
      #plt.plot(lbol_final,10**corr_final) #combination + low and high luminosity
      #plt.yscale('log');

      corr_fac = interpolate.interp1d(lbol_final,corr_final)
      luminosity=l_bol-corr_fac(l_bol)

    elif bol_corr == 'Duras20':
       # table1, (Klbol),general
       pars = [10.96, 11.93, 17.79]
       k_corr = np.log10(pars[0]*(1+((l_bol - 33.485)/pars[1])**pars[2]))
       luminosity=l_bol-k_corr
    else:
       assert False, "Bolometric correction is unknown"

    """
    plt.figure()
    plt.hist(luminosity,density=True)
    plt.ylabel('luminosity (2-10kev)')
    plt.show()
    """
    return luminosity, np.log10(lambda_char[-1]), lg_edd

def weightedFunction(parameter, duty_cycle, bins, volume):
    """ Function to calculate the
    :param parameter: (numpy) array, the varaible in question [variable units]
    :param duty_cycle: (numpy) array, the duty cycle [dimensionless]
    :param bins: (numpy) array, an evenly spaced array of bins for the main variables [variable units]
    :param volume: float, the volume of the cosmological volume
    :return:
    """

    if not utl.evenly_spaced(bins):
        raise ValueError("bins are not evenly spaced")
    if volume == 0.0:
        raise ValueError("Volume is zero, cannot calculate function with zero volume")

    step = abs(bins[1] - bins[0])
    lum_bins = stats.binned_statistic(parameter, duty_cycle, "sum", bins=bins)[0]
    lum_func = (lum_bins/volume)/step
    return lum_func

"""
    # Save Eddington Distribution Data
    step = 0.5
    lg_edd_derived = np.log10(25) + luminosity - (35.3802 + stellar_mass - 0.15)  # log10(1.26e38 * 0.002) = 35.3802
    edd_bin = np.arange(-4, 1, step)
    prob_derived = stats.binned_statistic(lg_edd_derived, duty_cycle, 'sum', bins=edd_bin)[0] / (step * sum(duty_cycle))
    edd_bin = edd_bin[:-1]
    edd_bin = edd_bin[prob_derived > 0]
    prob_derived = prob_derived[prob_derived > 0]
    edd_plotting_data = (utl.PlottingData(edd_bin, np.log10(prob_derived)))
    return luminosity, lg_edd, xlf_plotting_data, edd_plotting_data
"""

def generate_nh_distribution(lg_luminosity, z, lg_nh):
    """ Function written by Viola to generate (I think) a distribution of nh values for the appropriate luminosity.
    :param lg_luminosity: float, value for luminosity [log10 ]
    :param z: float, redshift
    :param lg_nh: float, array of possible nh values
    :return: array, the probability distribution over lg_nh.
    """
    xi_min, xi_max, xi0_4375, a1, eps, fctk, beta = 0.2, 0.84, 0.43, 0.48, 1.7, 1., 0.24
    if z > 2:
        z = 2
    xi_4375 = (xi0_4375*(1+z)**a1) - beta*(lg_luminosity - 43.75)
    max_ = max(xi_4375, xi_min)
    xi = min(xi_max, max_)
    fra = (1 + eps)/(3 + eps)
    f = np.ones(len(lg_nh))

    # Boolean flags to separate conditions
    flag = np.where((lg_nh < 21) & (lg_nh >= 20))
    flag1 = np.where((lg_nh < 22) & (lg_nh >= 21))
    flag2 = np.where((lg_nh < 23) & (lg_nh >= 22))
    flag3 = np.where((lg_nh < 24) & (lg_nh >= 23))
    flag4 = np.where((lg_nh < 26) & (lg_nh >= 24))
    if xi < fra:
        f[flag] = 1 - ((2 + eps)/(1 + eps))*xi
        f[flag1] = (1/(1 + eps))*xi
        f[flag2] = (1/(1 + eps))*xi
        f[flag3] = (eps/(1 + eps))*xi
        f[flag4] = (fctk/2)*xi
    else:
        f[flag] = (2/3) - ((3 + 2*eps)/(3 + 3*eps))*xi
        f[flag1] = (1/3) - (eps/(3 + 3*eps))*xi
        f[flag2] = (1/(1 + eps))*xi
        f[flag3] = (eps/(1 + eps))*xi
        f[flag4] = (fctk/2)*xi
    return f

def generate_nh_value_robust(index, length, nh_bins, lg_lx_bins, z):
    nh_distribution = (generate_nh_distribution(lg_lx_bins[index], z, nh_bins)) * 0.01  # call fn
    cum_nh_distribution = np.cumsum(nh_distribution[::-1])[::-1]
    norm_cum_nh_distribution = (cum_nh_distribution/cum_nh_distribution[0])[::-1]
    reverse_nh_bins = nh_bins[::-1]  # Reverse
    sample = np.random.random(length)
    interpolator = interpolate.interp1d(norm_cum_nh_distribution, reverse_nh_bins, bounds_error=False,
                                           fill_value=(reverse_nh_bins[0], reverse_nh_bins[-1]))
    return interpolator(sample)

def batch_nh(indexes, values, nh_bins, lg_lx_bins, z):
    out = []
    for index in indexes:
        flag = np.where(index == values)[0]
        if len(flag) == 0:
            pass
        else:
            component = (flag, generate_nh_value_robust(index, len(values[flag]),  nh_bins, lg_lx_bins, z))
            out.append(component)
    return out

def luminosity_to_nh(luminosity, z, parallel=True):
    """ Function to generate nh values for the AGN based on the luminosity.
    :param luminosity: array, the luminosity of the AGNs [log10 erg s^-1]
    :param z: float, redshift [dimensionless]
    :return: (numpy) array, the nh values [cm^-2]
    """
    lg_nh_range = np.arange(20., 30., 0.0001)  # Range, or effective 'possible values' of Nh
    nh = np.ones(len(luminosity))
    lg_lx_bins = np.arange(np.amin(luminosity), np.amax(luminosity), 0.01)
    bin_indexes = np.digitize(luminosity, lg_lx_bins)

    def generate_nh_value(index, length, nh_bins=lg_nh_range):
        """Internal function to generate an Nh value, just exists for parallelizing, below"""
        nh_distribution = (generate_nh_distribution(lg_lx_bins[index], z, nh_bins)) * 0.01  # call fn
        cum_nh_distribution = np.cumsum(nh_distribution[::-1])[::-1]
        norm_cum_nh_distribution = (cum_nh_distribution/cum_nh_distribution[0])[::-1]
        reverse_nh_bins = nh_bins[::-1]  # Reverse
        sample = np.random.random(length)
        
        interpolator = interpolate.interp1d(norm_cum_nh_distribution, reverse_nh_bins, bounds_error=False,
                                               fill_value=(reverse_nh_bins[0], reverse_nh_bins[-1]))
        return interpolator(sample)

    if not parallel:
        # This loop has been sped up but is still a bottleneck
        for i in range(len(lg_lx_bins) - 1):  # index for Lx bins
            # Serial
            flag = np.where(bin_indexes == i)
            nh[flag] = generate_nh_value(i, len(nh[flag]))

    if parallel:
        no_proc = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(no_proc)
        indexes_list = np.array_split(np.arange(len(lg_lx_bins) - 1), no_proc)
        res = [pool.apply_async(batch_nh, (indexes, bin_indexes, lg_nh_range, lg_lx_bins, z)) for indexes in indexes_list]
        results = [r.get() for r in res]
        pool.close()
        pool.join()
        continuous_results = []
        for result in results:
            continuous_results += result
        for element in continuous_results:
                nh[element[0]] = element[1]
    return nh

def nh_to_type(nh):
    """ Simple function that takes an nh value and assigns the AGN type.
    :param nh: array, the nh value
    :return: array, the type
    """
    type1 = np.ones_like(nh)
    type2 = np.ones_like(nh) * 2
    thick = np.ones_like(nh) * 3

    type = np.zeros_like(nh)
    type[nh < 22] = type1[nh < 22]
    type[(nh >= 22) * (nh < 24)] = type2[[(nh >= 22) * (nh < 24)]]
    type[nh >= 24] = thick[nh >= 24]
    return type

def compute_wp(x, y, z, period, weights=None, bins=(-1, 1.5, 50), pimax=50, threads="system"):
    """ Function to encapsulate wp from Corrfunc.
    :param x: array, x coordinate
    :param y: array, y coordinate
    :param z: array, z coordinate
    :param period: float, the axis of the cosmological volume
    :param weights: array, weights (if any)
    :param bins: tuple, of form (low, high, number of steps), representing the bins for WP. Values are in log10.
    :param pi_max: float, the value of pi_max
    :param threads: int/string, the number of threads to spawn. Setting threads="system" will spawn a number of threads
    equal to the (available) cores on your machine. Exceeding this will not result in performance increase.
    :return: PlottingData object, with x as the bins and y as xi.
    """
    if threads == "System" or threads == "system":
        threads = multiprocessing.cpu_count()
    r_bins = np.logspace(bins[0], bins[1], bins[2])

    print("Weights, max = {}. , min = {}".format(np.amax(weights), np.amin(weights)))


    wp_results = wp(period, pimax, threads, r_bins, x, y, z, weights=weights, weight_type='pair_product', verbose=True)
    xi = wp_results['wp']
    return utl.PlottingData(r_bins[:-1], xi)

def compute_bias(variable, parent_halo_mass, z, h, cosmology, bin_size = 0.3, weight=None, mask=None):
    """ Function to compute the bias for a supplied variable. Viola wrote much of this function.
    :param variable: array, the variable to compute the bias against (log10)
    :param parent_halo_mass: array, the parent halo mass
    :param z: float, redshift
    :param h: float, reduced hubble constant
    :param cosmology: Corrfunc cosmology object, storing all the cosmological parameters.
    :param bin_size: float, size of the bins (log10) - bin high and low values are automatically calculated.
    :param weight: array, weights to the bias
    :param mask: array, if desired, we can mask the data
    :return: PlottingData object, with the bias vs the bins.
    """
    def func_g_squared(z):
        matter = cosmology.Om0 * (1 + z) ** 3
        curvature = (1 - cosmology.Om0 - cosmology.Ode0) * (1 + z) ** 2
        return matter + curvature + cosmology.Ode0

    def func_d(omega, omega_l):
        # Helper functions
        a = 5 * omega / 2
        b = omega ** (4 / 7.) - omega_l
        c = 1 + omega / 2
        d = 1 + omega_l / 70.
        d = a / (b + c * d)
        return d

    def func_omega_l(omega_l0, g_squared):
        # Eisenstein 1999
        return omega_l0 / g_squared

    def func_delta_c(delta_0_critical, d):
        # delta_c as a function of omega and linear growth
        # van den Bosch 2001
        return delta_0_critical/d

    def func_delta_0_critical(omega, p):
        # A3 van den Bosch 2001
        return 0.15 * (12 * 3.14159) ** (2 / 3.) * omega ** p

    def func_p(omega_m0, omega_l0):
        # A4 van den Bosch 2001
        if omega_m0 < 1 and omega_l0 == 0:
            return 0.0185
        if omega_m0 + omega_l0 == 1.0:
            return 0.0055
        else:
            return 0.0055  # VIOLA, had to add this in to make it work with my cosmology, I assume this is okay?

    def func_omega(z, omega_m0, g_squared):
        # A5 van den Bosch 2001 / eq 10 Eisenstein 1999
        return omega_m0 * (1 + z) ** 3 / g_squared

    def func_sigma(sigma_8, f, f_8):
        # A8 van den Bosch 2001
        return sigma_8 * f / f_8

    def func_u_8(gamma):
        # A9 van den Bosch 2001
        return 32 * gamma

    def func_u(gamma, M):
        # A9 van den Bosch 2001
        return 3.804e-4 * gamma * (M / cosmology.Om0) ** (1 / 3.)

    def func_f(u):
        # A10 van den Bosch 2001
        common = 64.087
        factors = (1, 1.074, -1.581, 0.954, -0.185)
        exps = (0, 0.3, 0.4, 0.5, 0.6)

        ret_val = 0.0
        for i in range(len(factors)):
            ret_val += factors[i] * u ** exps[i]

        return common * ret_val ** (-10)

    def func_b_eul(nu, delta_sc=1.686, a=0.707, b=0.5, c=0.6):
        # eq. 8 Sheth 2001
        a = np.sqrt(a) * delta_sc
        b = np.sqrt(a) * a * nu ** 2
        c = np.sqrt(a) * b * (a * nu ** 2) ** (1 - c)
        d = (a * nu ** 2) ** c
        e = (a * nu ** 2) ** c + b * (1 - c) * (1 - c / 2)
        return 1 + (b + c - d / e) / a

    def func_b_eul_tin(nu, delta_sc=1.686, a=0.707, b=0.35, c=0.8):
        # eq. 8 Tinker 2005
        a = np.sqrt(a) * delta_sc
        b = np.sqrt(a) * a * nu ** 2
        c = np.sqrt(a) * b * (a * nu ** 2) ** (1 - c)
        d = (a * nu ** 2) ** c
        e = (a * nu ** 2) ** c + b * (1 - c) * (1 - c / 2)
        return 1 + (b + c - d / e) / a

    def estimate_sigma(M, z, g_squared, omega_m0=cosmology.Om0, gamma=0.2, sigma_8=0.8):
        # Estimate sigma for a set of masses
        # vdb A9
        u = func_u(gamma, M)
        u_8 = func_u_8(gamma)
        # vdb A10
        f = func_f(u)
        f_8 = func_f(u_8)
        # vdb A8
        sigma = func_sigma(sigma_8, f, f_8)
        return sigma

    def estimate_delta_c(M, z, g_squared, gamma=0.2, omega_m0=cosmology.Om0, omega_L0=cosmology.Ode0):
        # Estimate delta_c for a set of masses
        # Redshift/model dependant parameters
        omega = func_omega(z, omega_m0, g_squared)
        omega_l = func_omega_l(omega_L0, g_squared)
        # vdb A3
        p = func_p(omega_m0, omega_L0)
        # Allevato code
        d1 = func_d(omega, omega_l) / (1 + z)
        d0 = func_d(omega_m0, omega_L0)
        d = d1 / d0
        delta_0_crit = func_delta_0_critical(omega, p)
        delta_0_crit = 1.686
        delta_c = func_delta_c(delta_0_crit, d)
        return delta_c, delta_0_crit

    def estimate_bias_tin(m, z, g_squared, gamma=0.2, omega_m0=cosmology.Om0, omega_L0=cosmology.Ode0,
                          sigma_8=0.8):
        # Estimate the bias Tinker + 2005
        sigma = estimate_sigma(m, z, g_squared, omega_m0, gamma, sigma_8)
        delta_c, delta_0_crit = estimate_delta_c(m, z, g_squared, gamma, omega_m0, omega_L0)
        nu = delta_c / sigma
        return func_b_eul_tin(nu, delta_0_crit)

    g_squared = func_g_squared(z)
    bias = estimate_bias_tin(((10**parent_halo_mass) * 0.974) * h, z, g_squared)
    bins = np.arange(np.amin(variable), np.amax(variable), bin_size)
    mean_bias = np.zeros_like(bins)
    error_bias = np.zeros_like(bins)

    for i in range(len(bins) - 1):
        n1 = np.where(((variable) >= bins[i]) & (variable < bins[i + 1]))
        if mask is not None:
            n1 *= mask  # Mask out for obscured etc if we want to.
        if weight is not None:
            mean_bias[i] = np.sum(bias[n1] * weight[n1]) / np.sum(weight[n1])
            error_bias[i] = np.sqrt((np.sum(weight[n1] * (bias[n1] - mean_bias[i]) ** 2)) / (
                        ((len(weight[n1]) - 1) / len(weight[n1])) * np.sum(weight[n1])))
        else:
            mean_bias[i] = np.mean(bias[n1])
            error_bias[i] = np.std(bias[n1])
    return utl.PlottingData(bins[0:-1], mean_bias[0:-1], error_bias[0:-1])

def calculate_hod(up_id, halo_mass, duty_cycle_weight, centrals=True):
    """ Function to estimate the HOD of a catalogue, only centrals.
    :param up_id: array, the up_id of a cataloguea
    :param halo_mass: array, the halo mass (log10)
    :param duty_cycle_weight: array, the duty cycle (for weighting)
    :param centrals: bool, flag to turn off only calculating for centrals, to calculate for all galaxies.
    :return:
    """
    flagCentrals = np.where(up_id > 0)  # Centrals

    #Halo_mass = np.log10(self.mvir)

    bins = np.arange(11, 15, 0.1)
    bins_out = bins[0:-1]

    hist_centrals_unweighted = np.histogram(halo_mass[flagCentrals], bins)[0]
    hist_all = np.histogram(halo_mass, bins)[0]

    if centrals:
        flag = flagCentrals
    elif not centrals:
        flag = np.invert(flagCentrals)
    else:
        assert False, "Invalid Value for centrals, should be Boolean"

    if duty_cycle_weight is not None:
        hist_subject = np.histogram(halo_mass[flag], bins, weights=duty_cycle_weight[flag])[0]
    else:
        hist_subject = np.histogram(halo_mass[flag], bins)[0]

    t = np.where(bins_out <= 11.7)  # Not quite sure what these are, or what they are for?
    h = np.where(bins_out > 11.2)
    l = np.where(bins_out <= 11.2)

    hod = np.zeros_like(bins_out)

    if not centrals:  # Not sure why we are doing this?
        hod[t] = 0.0001
        hod = hist_subject / hist_centrals_unweighted
    else:
        hod[h] = hist_subject[h] / hist_centrals_unweighted[h]
        hod[l] = hist_subject[l] / hist_all[l]

    return utl.PlottingData(bins[0:-1], hod)

def SFR(z,Mstar, method='Tomczak16'):
    # returns SFR in log scale
    sig = 0.2 # intrinsic scatter in the relation in dex

    if method == 'Tomczak16':
        #adopt Tomczak+16
        s0=0.195+1.157*z-0.143*z**2.
        gam=1.118
        M0=9.244+0.753*z-0.090*z**2.
        return np.log10(10.**(s0-np.log10(1.+(10.**(Mstar-M0))**(-gam))))+np.random.normal(0.,sig,len(Mstar))

    elif method == "Schreiber15":
        r = np.log10(1+z)
        m = Mstar-9
        pars = np.array([1.5, 0.3, 2.5, 0.5, 0.36])
        return schreiber(m, r, *pars) + np.random.normal(0., sig, len(Mstar))

    elif method == "Carraro20":
        r = np.log10(1+z)
        m = Mstar-9
        pars = np.array([2.29082663, 0.25278264, 0.33402409, 0.63679373, 0.54958314])
        return schreiber(m, r, *pars) + np.random.normal(0., sig, len(Mstar))

    else:
        assert False, "Method is unknown"

def SFR_Q(z,Mstar):
   # returns SFR in log scale
   assert (z==0.45 or z == 1.0 or z==2.7),"Entered a wrong redshift value. SFR currently defined for z=0.45, z=1.0 or z=2.7."
   sig = 0.2 # intrinsic scatter in the relation in dex
   if z==0.45:
      return (0.40 *Mstar-4.5)+np.random.normal(0.,sig,len(Mstar))
   if z==1.:
      return (0.37 *Mstar-3.6)+np.random.normal(0.,sig,len(Mstar))
   if z==2.7:
      a=np.empty(Mstar.shape)
      a.fill(np.nan)
      return a

def SFR_SB(z,Mstar):
   # returns SFR in log scale
   assert (z==0.45 or z == 1.0 or z==2.7),"Entered a wrong redshift value. SFR currently defined for z=0.45, z=1.0 or z=2.7."
   sig = 0.2 # intrinsic scatter in the relation in dex
   if z==0.45:
      return (0.73 *Mstar-5.8)+np.random.normal(0.,sig,len(Mstar))
   if z==1.:
      return (0.61 *Mstar-4.3)+np.random.normal(0.,sig,len(Mstar))
   if z==2.7:
      return (0.59 *Mstar-3.4)+np.random.normal(0.,sig,len(Mstar))

def schreiber(m,r,a0,a1,a2,m0,m1):
    return m - m0 + a0*r - a1*(np.maximum(0.,m-m1-a2*r))**2

def lusso(x, a1, a2, a3, b):
   return a1*x + a2*x**2 + a3*x**3 + b

if __name__ == "__main__":
    cosmo = 'planck18'
    cosmology = cosmology.setCosmology(cosmo)
    volume = 200**3

    halos = generate_semi_analytic_halo_catalogue(volume, (12, 16, 0.1), 0, 0.7,
                                                  visual_debugging=False,
                                                  erase_debugging_folder=True,
                                                  visual_debugging_path="./visualValidation/SemiAnalyticCatalog/")
    stellar_mass = halo_mass_to_stellar_mass(halos, 0,
                                             visual_debugging=False,
                                             erase_debugging_folder=True,
                                             debugging_volume=volume,
                                             visual_debugging_path="./visualValidation/StellarMass/")

    black_hole_mass = stellar_mass_to_black_hole_mass(stellar_mass,
                                                      method="Shankar16",
                                                      scatter="Intrinsic",
                                                      visual_debugging=False,
                                                      erase_debugging_folder=True,
                                                      debugging_volume=volume,
                                                      visual_debugging_path="./visualValidation/BlackHoleMass/")

    duty_cycle = to_duty_cycle("Geo", stellar_mass, black_hole_mass, 0)

    luminosity = black_hole_mass_to_luminosity(black_hole_mass, duty_cycle, stellar_mass, 0)

    nh = luminosity_to_nh(luminosity, 0)
    agn_type = nh_to_type(nh)