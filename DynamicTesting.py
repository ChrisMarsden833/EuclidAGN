# Libraries
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import stats
from scipy import special
import pandas as pd
import os
import re
import multiprocessing
from numba import jit
from math import pi
import time

# Specific Libraries
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import Corrfunc
from Corrfunc.theory import wp
from Corrfunc.theory import DD

# Local
import AGNCatalogToolbox as at
from ACTUtillity import *
from ImageGeneration import *


class EuclidMaster:
    """Class that encapsulates the Euclid code.

    This object should be created, and then it's member functions should be
    called in the appropriate order with varying parameters where desired. The
    'getter' functions can then be used to extract relavant data.

    Attributes:
        cosmo (string): A string describing the cosmology the class will use.
            The default is planck18. This is passed straight to
            colossus.cosmology, so define as appropriate.
    """

    def __init__(self, cosmo='planck18'):
        self.cosmology_name = cosmo
        self.cosmology = cosmology.setCosmology(cosmo)

        # Extract Reduced Hubble Constant, because we use it so much
        self.h = self.cosmology.H0/100

        # Image and data file location
        self.path_visual_validation = ValidatePath("./visualValidation/")
        self.path_data = ValidatePath("./Data/", ErrorOnFail=True)
        self.path_bigData = ValidatePath("./BigData/", ErrorOnFail=True)

        # Initialization (not strictly necessary but for PEP8 compliance...
        self.z = 0
        self.dm_type = 'Undefined'
        self.volume_axis = 0
        self.volume = 0

        # Primary Catalog
        self.main_catalog = []

        self.XLF_plottingData = []
        self.Edd_plottingData = []
        self.WP_plottingData = []
        self.bias_plottingData = []
        self.HOD_plottingData = []

        self.process_bookkeeping = {
            "Dark Matter": False,
            "Stellar Mass": False,
            "Black Hole Mass": False,
            "Eddington Ratios": False,
            "Duty Cycle": False}

    def set_z(self, z=0.):
        self.z = z

    def define_main_catalog(self, length):
        dt = [("x", np.float32),
              ("y", np.float32),
              ("z", np.float32),
              ("effective_halo_mass", np.float32),
              ("effective_z", np.float32),
              ("stellar_mass", np.float32),
              ("black_hole_mass", np.float32),
              ("duty_cycle", np.float32),
              ("luminosity", np.float32)]
        self.main_catalog = np.zeros(length, dtype=dt)
        self.main_catalog['effective_z'] = np.ones(length) * self.z

    def load_dm_catalogue(self, generate_figures=True, filename="MD_"):
        """ Function to load in the catalog_data from the multi-dark halo catalogue

        This catalog_data should exist as .npy files in the Directory/BigData. Within
        this folder there should be a script to pull these out of the SQL
        database. Note that this expects a .npy file in the with columns x, y, z
        scale factor at accretion, mass at accretion. If generateFigures is set
        to True (default), then a further column of the halo mass is also
        required to validate the halos.

        Attributes:
            generate_figures (bool) : flag to generate the figures that exist for
                visual validation. Defaults to true.
            filename (string) : component of the filename excluding the redshift
                - the closest z will be found automatically. Default is "MD_",
                expecting files of the form "MD_0.0.npy".
        """
        print("Loading Halo Catalogue")
        self.dm_type = "N-Body"  # Flag used for later
        self.volume_axis = 1000 / self.h
        self.volume = self.volume_axis**3  # MultiDark Box size, Mpc

        # Search for the closest file and read it in.
        catalog_file, catalog_z = GetCorrectFile(filename, self.z, self.path_bigData, True)
        print("Found file:", catalog_file)
        catalog_data = np.load(self.path_bigData + catalog_file)
        print("dtypes found: ", catalog_data.dtype)

        self.define_main_catalog(len(catalog_data))

        # If we want to generate the Halo Mass function figure, this section.
        if generate_figures:
            PlotHaloMassFunction(catalog_data["mvir"][catalog_data["upid"] == -1] / self.h, self.z, self.volume,
                                 self.cosmology, self.path_visual_validation)
        
        # Store the catalog_data in class variable, correcting for h.
        self.main_catalog['x'] = catalog_data['x']/self.h
        self.main_catalog['y'] = catalog_data['y']/self.h
        self.main_catalog['z'] = catalog_data['z']/self.h

        # These things don't need to go into the class (yet), as we're about to use them.
        main_id = catalog_data["id"]
        up_id = catalog_data["upid"]
        virial_mass = catalog_data["mvir"]/self.h
        mass_at_accretion = catalog_data["Macc"]/self.h
        accretion_scale = catalog_data["Acc_Scale"]

        del catalog_data  # Save on memory

        # Reserved memory
        virial_mass_parent = np.zeros_like(virial_mass)
        idh = np.zeros_like(virial_mass)
        effective_z = np.zeros_like(accretion_scale)

        print('    Sorting list w.r.t. upId')
        sorted_indexes = up_id.argsort()  # + 1 Array, rest are reused
        # To maintain order, we update the class data. This only needs to be done once.
        self.main_catalog['x'] = self.main_catalog['x'][sorted_indexes]
        self.main_catalog['y'] = self.main_catalog['y'][sorted_indexes]
        self.main_catalog['z'] = self.main_catalog['z'][sorted_indexes]
        # We also do this with the local data
        main_id = main_id[sorted_indexes]
        up_id = up_id[sorted_indexes]
        virial_mass = virial_mass[sorted_indexes]
        mass_at_accretion = mass_at_accretion[sorted_indexes]
        accretion_scale = accretion_scale[sorted_indexes]
        up_id_0 = np.searchsorted(up_id, 0)  # Position where zero should sit on the now sorted up_id.
        # All arrays are should now sorted by up_id.

        print('    copying all {} elements with up_id = -1'.format(str(up_id_0)))
        virial_mass_parent[:up_id_0] = virial_mass[:up_id_0]  # MVir of centrals, or where up_id  = -1.

        print('    sorting remaining list list w.r.t. main id')
        up_id_cut = up_id[up_id_0:]  # Up_id's that are not -1, or the satellites, value pointing to their progenitor.

        id_cut = main_id[:up_id_0]  # ids of centrals
        virial_mass_cut = virial_mass[:up_id_0]  # masses of centrals

        sorted_indexes = id_cut.argsort()  # get indexes to centrals by id.
        id_cut = id_cut[sorted_indexes]  # actually sort centrals by id.
        virial_mass_cut = virial_mass_cut[sorted_indexes]  # sort virial masses the same way.

        print('    copying remaining', str(len(up_id) - up_id_0), 'elements')
        sorted_indexes = np.searchsorted(id_cut, up_id_cut)  # indexes of where satellite id's point to centrals
        virial_mass_parent[up_id_0:] = virial_mass_cut[sorted_indexes]  # Sort parents by this, and assign to satellites
        # This gives us the virial mass of the parent or itself if it is a parent. But do we actually need this?
        idh[up_id_0:] = 1

        halo_mass = virial_mass
        halo_mass[idh > 0] = mass_at_accretion[idh > 0]

        effective_z[idh > 0] = 1/accretion_scale[idh > 0] - 1
        effective_z[idh < 1] = catalog_z

        # self.main_catalog['virial_mass'] = virial_mass # Not sure we actually need this?

        self.main_catalog['effective_halo_mass'] = np.log10(halo_mass)
        self.main_catalog['effective_z'] = effective_z
        # self.main_catalog['parent_halo_mass'] = np.log10(virial_mass_parent)
        # self.main_catalog['up_id'] = up_id

    def generate_semi_analytic_halos(self, volume=500**3, mass_low=12., mass_high=16., generate_figures=True):
        """Function to generate a catalog of semi-analytic halos.

        Function to pull a catalogue of halos from the halo mass function. A
        reasonable volume should be chosen - a larger volume will of course
        produce a greater number of halos, which will increase resolution at
        additional computational expense.

        Attributes:
            volume (float) : The volume of the region within which we will
                create the halos.
            mass_low (float) : The lowest mass (in log10 M_sun) halo to generate
                defaulting to 11.
            mass_high (float) : The highest mass halo to generate, defaulting to
                15
            generate_figures (bool) : flag to generate the figures that exist for
                visual validation. Defaults to true.
        """
        print("Generating Semi-Analytic halos")

        self.dm_type = "Analytic"
        self.volume = volume

        temporary_halos = \
            at.generate_semi_analytic_halo_catalogue(catalogue_volume=volume,
                                                     mass_params=(mass_low, mass_high, 0.1),
                                                     z=self.z,
                                                     h=self.h,
                                                     visual_debugging=False,
                                                     erase_debugging_folder=True,
                                                     visual_debugging_path="./visualValidation/SemiAnalyticCatalog/")
        self.define_main_catalog(len(temporary_halos))
        self.main_catalog['effective_halo_mass'] = temporary_halos

    def assign_stellar_mass(self, formula="Grylls18", scatter=0.001):
        """ Function to generate stellar masses from halo masses

        Just calls the 'class free' function from galaxy_physics

        :param formula: string, the method to use. Options currently include "Grylls18" and "Moster"
        :param scatter: float, the magnitude of the scatter (in dex).
        :return: None
        """
        print("Assigning Stellar Mass")
        self.main_catalog['stellar_mass'] = at.halo_mass_to_stellar_mass(self.main_catalog['effective_halo_mass'],
                                                                         self.main_catalog['effective_z'],
                                                                         formula=formula,
                                                                         scatter=scatter,
                                                                         visual_debugging=False,
                                                                         erase_debugging_folder=True,
                                                                         debugging_volume=self.volume,
                                                                         visual_debugging_path=
                                                                         "./visualValidation/StellarMass/")

    def assign_black_hole_mass(self, formula="Shankar16", scatter="Intrinsic"):
        """Function to generate black hole masses atop stellar masses.

        :param formula: string, specifying the method to be used, options are "Shankar16",  "KormondyHo" and "Eq4".
        :param scatter: string or float, string should be "Intrinsic", float value specifies the (fixed) scatter magnitude
        :return: None
        """
        print("Assigning Black Hole Mass")
        self.main_catalog['black_hole_mass'] = at.stellar_mass_to_black_hole_mass(self.main_catalog['stellar_mass'],
                                                                                  method=formula,
                                                                                  scatter=scatter,
                                                                                  visual_debugging=False,
                                                                                  erase_debugging_folder=True,
                                                                                  debugging_volume=self.volume,
                                                                                  visual_debugging_path=
                                                                                  "./visualValidation/BlackHoleMass/")

    def assign_duty_cycle(self, function="Mann"):
        """Function to assign black hole masses atop stellar masses.

        :param function: string/float, string specifying the method, options are "Mann"/"Schulze", or a constant float
        :return: None
        """
        print("Assigning Duty Cycle, using {}'s method".format(function))
        self.main_catalog["duty_cycle"] = at.to_duty_cycle(function,
                                                           self.main_catalog['stellar_mass'],
                                                           self.main_catalog['black_hole_mass'],
                                                           self.z)

    def assign_luminosity(self, method="Schechter", redshift_evolution=False, parameter1 = -1, parameter2 = -0.65):
        """ Function to assign luminosity (based on black hole mass)

        :param method: string, the function to pull the Eddington Ratios from. Options are "Schechter", "PowerLaw" or
        "Gaussian".
        :param redshift_evolution: bool, if set to true will introduce a factor representing the z-evolution.
        :param parameter1: float, the first parameter for the method. For Schechter it is the knee, for PowerLaw it is not used,
        and for the Gaussian it is sigma.
        :param parameter2: float, the second parameter for the method. For Schechter it is alpha, for PowerLaw it is not used, for
        Gaussian it is b.
        :return: None
        """
        print("Assigning Luminosity")
        self.main_catalog['luminosity'], xlf_plotting_data, edd_plotting_data = \
            at.black_hole_mass_to_luminosity(self.main_catalog["black_hole_mass"],
                                             self.main_catalog["duty_cycle"],
                                             self.main_catalog["stellar_mass"],
                                             self.z,
                                             method=method,
                                             redshift_evolution=redshift_evolution,
                                             parameter1=parameter1,
                                             parameter2=parameter2,
                                             return_plotting_data=True,
                                             volume=self.volume)
        # Store the plotting data in class for later comparison.
        self.XLF_plottingData.append(xlf_plotting_data)
        self.Edd_plottingData.append(edd_plotting_data)


    def CreateCatalogue(self, LumCut = True, LumLow = 42.01, LumHigh = 44.98):
        print("Creating Catalogue")
        """Function to create a catalogue by assigning the duty cycle.

        This function takes the duty cycle and using it as a weight, will create
        a weighted random cataloge.

        Attributes:
            LumCut (bool) : flag to switch off the luminosity cut, defaults to True.
            LumLow (float) : The low limit of the luminosity cut (in log10), defaults to 42.01.
            LumHigh (float) : The high limit of the luminosity cut (in log10), defaults to 44.98
        """
        TestForRequiredVariables(self, ["dutycycle", "luminosity", "x_coord", "y_coord", "z_coord"])
        # Random allocation
        random = np.random.rand(len(self.dutycycle))
        dutyCycleFlag = random < self.dutycycle

        if LumCut:
            lum_flag = (self.luminosity >= LumLow) * (self.luminosity <= LumHigh)
        else:
            lum_flag = 1

        print("    {}% of Galaxies remain as AGN".format(np.round(100 * np.sum(dutyCycleFlag * lum_flag)/len(self.x_coord), 2)))

        self.x_cat = self.x_coord[dutyCycleFlag * lum_flag]
        self.y_cat = self.y_coord[dutyCycleFlag * lum_flag]
        self.z_cat = self.z_coord[dutyCycleFlag * lum_flag]
        self.lum_cat = self.luminosity[dutyCycleFlag * lum_flag]
        self.dc_cat = self.dutycycle[dutyCycleFlag * lum_flag]

    def Obscuration(self):
        print("Calculating Obscuration")
        """Function to assign column density, Nh,

        Also assigns the .type variable which contains the agn type (1, 2 or 3 (3 is obscured))

        No attributes
        """

        TestForRequiredVariables(self, ["luminosity"])

        lgNH = np.arange(20., 30., 0.0001)  # Range, or effective 'possible values' of Nh

        lgNHAGNSch1 = np.ones(len( self.luminosity))

        lgLxbin = np.arange(np.amin(self.luminosity), np.amax(self.luminosity), 0.01)

        bin_indexes = np.digitize(self.luminosity, lgLxbin)

        def GenerateNHValue(i, length, lgNH = lgNH):
            fLxzNH = (NHfunc(lgLxbin[i], self.z, lgNH))  # call fn
            fLxzNH = fLxzNH * 0.01
            fLxzNH_ = fLxzNH[::-1]  # reverse
            fLxzNH_cum = np.cumsum(fLxzNH_)  # cumulative sum
            r_fLxzNH_cum = fLxzNH_cum[::-1]  # Reverse again
            y = r_fLxzNH_cum / r_fLxzNH_cum[0]  # Normalize(ish?) by the first element
            y = y[::-1]  # Reverse AGAIN
            lgNH2 = lgNH[::-1]  # Reverse
            a = np.random.random(length)
            return np.interp(a, y, lgNH2, right=-99)

        #flagger = 0
        #NHer = 0

        #start = time.process_time()

        # This loop is super slow - TODO need to speed this up.
        for i in range(len(lgLxbin)-1): # index for Lx bins

            #flag_start = time.process_time()
            ######################################
            flag = np.where(bin_indexes == i)
            ######################################
            #flag_stop = time.process_time()

            #flagger += abs(flag_start - flag_stop)

            #NhStart = time.process_time()

            ######################################
            lgNHAGNSch1[flag] = GenerateNHValue(i, len(lgNHAGNSch1[flag]))
            ######################################
            #NHstop = time.process_time()

            #NHer += abs(NhStart - NHstop)

        #print(time.process_time() - start)

        #print("Flag:", flagger)
        #print("NHer:", NHer)

        self.N_h = lgNHAGNSch1
        
        N_h = lgNHAGNSch1

        type1 = np.ones_like(N_h)
        type2 = np.ones_like(N_h) * 2
        thick = np.ones_like(N_h) * 3

        in_type = np.zeros_like(N_h)
        in_type[N_h < 22] = type1[N_h < 22]
        in_type[(N_h >= 22)*(N_h < 24)] = type2[[(N_h >= 22)*(N_h < 24)]]
        in_type[N_h >= 24] = thick[N_h >= 24]

        self.type = in_type


    def computeWP(self, threads = "System", pi_max = 50, binStart = -1, binStop = 1.5, binSteps = 50):
        print("Computing wp")
        """Function to compute the correlation function wp

        Attributes:
            threads (int/string) : The specifier for the number of threads to create.
                default is "System", which uses all available cores.
            pi_max (float) : The value of pi max for wp. Defaults to 50
            binStart (float) : Low limit to the (log spaced) bins. Defaults to 1
            binStop (float) : High limit to the (log spaced) bins. Defaults t0 1.5
            binSteps (int) : The number of spaces in the bins. Defaults to 50
        """
        TestForRequiredVariables(self, ["x_coord", "y_coord", "z_coord", "lum_cat", "dutycycle"])
        if threads == "System":
            threads = multiprocessing.cpu_count()
        period = self.volume**(1/3)
        rbins = np.logspace(-1, 1.5, 50)
        self.wpbins = rbins
        wp_results = wp(period, pi_max, threads, rbins,\
                self.x_coord, self.y_coord, self.z_coord, weights = self.dutycycle, weight_type = 'pair_product', verbose = True)
        xi = wp_results['wp']

        self.WP_plottingData.append(PlottingData(rbins[:-1], xi))

    def computeBias(self, variable, binsize = 0.3, weight = True, mask = None):
        # Functions borrowed from Viola's Code - I'm not entirely sure what they all do.
        requiredVariables = ["parent_halo_mass"]
        if weight:
            requiredVariables.append("dutycycle")
        TestForRequiredVariables(self, requiredVariables)

        def func_g_squared(z):
            matter = self.cosmology.Om0*(1+z)**3
            curvature = (1 - self.cosmology.Om0 - self.cosmology.Ode0)*(1+z)**2
            return matter + curvature + self.cosmology.Ode0

        def func_D(Omega, Omega_L):
            # Helper functions
            A = 5 * Omega / 2
            B = Omega**(4/7.) - Omega_L
            C = 1 + Omega / 2
            D = 1 + Omega_L / 70.
            D = A / (B + C * D)
            return D

        def func_Omega_L(Omega_L0, g_squared):
            # Eisenstein 1999
            return Omega_L0 / g_squared

        def func_delta_c(delta_0_crit, D):
            # delta_c as a function of omega and linear growth
            # van den Bosch 2001
            return delta_0_crit / D

        def func_delta_0_crit(Omega, p):
            # A3 van den Bosch 2001
            return 0.15 * (12*3.14159)**(2/3.) * Omega**p

        def func_p(Omega_m0, Omega_L0):
            # A4 van den Bosch 2001
            if (Omega_m0 < 1 and Omega_L0 == 0):
                return 0.0185
            if (Omega_m0 + Omega_L0 == 1.0):
                return 0.0055
            else:
                return 0.0055 # VIOLA, had to add this in to make it work with my cosmology, I assume this is okay?

        def func_Omega(z, Omega_m0, g_squared):
            # A5 van den Bosch 2001 / eq 10 Eisenstein 1999
            return Omega_m0 * (1+z)**3 / g_squared

        def func_sigma(sigma_8, f, f_8):
            # A8 van den Bosch 2001
            return sigma_8 * f / f_8

        def func_u_8(Gamma):
            # A9 van den Bosch 2001
            return 32 * Gamma

        def func_u(Gamma, M):
            # A9 van den Bosch 2001
            return 3.804e-4 * Gamma * (M/self.cosmology.Om0)**(1/3.)

        def func_f(u):
            # A10 van den Bosch 2001
            common = 64.087
            factors = (1, 1.074, -1.581, 0.954, -0.185)
            exps = (0, 0.3, 0.4, 0.5, 0.6)

            ret_val = 0.0
            for i in range(len(factors)):
                ret_val += factors[i] * u ** exps[i]

            return common * ret_val**(-10)

        def func_b_eul(nu, delta_sc=1.686, a=0.707, b=0.5, c=0.6):
            # eq. 8 Sheth 2001
            A = np.sqrt(a) * delta_sc
            B = np.sqrt(a) * a * nu**2
            C = np.sqrt(a) * b * (a * nu**2)**(1-c)
            D = (a * nu**2)**c
            E = (a*nu**2)**c + b*(1-c)*(1-c/2)
            return 1 + (B + C - D/E)/A

        def func_b_eulTin(nu, delta_sc=1.686, a=0.707, b=0.35, c=0.8):
            # eq. 8 Tinker 2005
            A = np.sqrt(a) * delta_sc
            B = np.sqrt(a) * a * nu**2
            C = np.sqrt(a) * b * (a * nu**2)**(1-c)
            D = (a * nu**2)**c
            E = (a*nu**2)**c + b*(1-c)*(1-c/2)

            return 1 + (B + C - D/E)/A

        def estimate_sigma(M, z, g_squared, Omega_m0 = self.cosmology.Om0, Gamma=0.2, sigma_8=0.8):
            # Estimate sigma for a set of masses
            # vdb A9
            u = func_u(Gamma, M)
            u_8 = func_u_8(Gamma)
            # vdb A10
            f = func_f(u)
            f_8 = func_f(u_8)
            # vdb A8
            sigma = func_sigma(sigma_8, f, f_8)
            return sigma

        def estimate_delta_c(M, z, g_squared, Gamma=0.2, Omega_m0 = self.cosmology.Om0, Omega_L0 = self.cosmology.Ode0):
            # Estimate delta_c for a set of masses
            # Redshift/model dependant parameters
            Omega = func_Omega(z, Omega_m0, g_squared)
            Omega_L = func_Omega_L(Omega_L0, g_squared)
            # vdb A3
            p = func_p(Omega_m0, Omega_L0)
            # Allevato code
            D1 = func_D(Omega, Omega_L) / (1 + z)
            D0 = func_D(Omega_m0, Omega_L0)
            D = D1/D0
            delta_0_crit = func_delta_0_crit(Omega, p)
            # TODO: remove
            delta_0_crit = 1.686
            delta_c = func_delta_c(delta_0_crit, D)
            return (delta_c, delta_0_crit)

        def estimate_biasTin(M, z, g_squared, Gamma=0.2, Omega_m0=self.cosmology.Om0, Omega_L0 = self.cosmology.Ode0, sigma_8 = 0.8):
            # Estimate the bias Tinker + 2005
            sigma = estimate_sigma(M, z, g_squared, Omega_m0, Gamma, sigma_8)
            delta_c, delta_0_crit = estimate_delta_c(M, z, g_squared, Gamma, Omega_m0, Omega_L0)
            nu = delta_c / sigma
            return func_b_eulTin(nu, delta_0_crit)

        g_squared = func_g_squared(self.z)
        bias = estimate_biasTin(((10.**self.parent_halo_mass)*0.974)*self.h, self.z, g_squared) # So this is the bias of the haloes?

        bin = np.arange(np.amin(variable), np.amax(variable), binsize)
        meanbias = np.zeros_like(bin)
        errorbias = np.zeros_like(bin)

        for i in range(len(bin)-1):
            N1 = np.where(((variable) >= bin[i]) & ((variable) < bin[i+1]))
            if mask != None:
                N1 *= mask # Mask out for obscured etc if we want to.
            if weight:
                meanbias[i] = np.sum(bias[N1]*self.dutycycle[N1])/np.sum(self.dutycycle[N1])
                errorbias[i] = np.sqrt((np.sum(self.dutycycle[N1] * (bias[N1] - meanbias[i])**2))/(((len(self.dutycycle[N1]) - 1)/len(self.dutycycle[N1])) * np.sum(self.dutycycle[N1])))
            else:
                meanbias[i] = np.mean(bias[N1])
                errorbias[i] = np.std(bias[N1])

        self.bias_plottingData.append(PlottingData(bin[0:-1], meanbias[0:-1], errorbias[0:-1]))

    def CalculateHOD(self, Centrals = True, weights = True):
        """Docstring
        """
        flagCentrals = np.where(self.upid > 0) # Centrals

        Halo_mass = np.log10(self.mvir)

        bins = np.arange(11, 15, 0.1)
        bins_out = bins[0:-1]

        hist_Centrals_unweighted = np.histogram(Halo_mass[flagCentrals], bins)[0]
        hist_all = np.histogram(Halo_mass, bins)[0]

        if Centrals:
            flag = flagCentrals
        elif not Centrals:
            flag = np.invert(flagCentrals)
        else:
            assert False, "Invalid Value for centrals, should be Boolean"

        if weights:
            hist_subject = np.histogram(Halo_mass[flag], bins, weights = self.dutycycle[flag])[0]
        elif not weights:
            hist_subject = np.histogram(Halo_mass[flag], bins)[0]
        else:
            assert False, "Invalud value for weights, should be Boolean"

        t = np.where(bins_out <= 11.7 ) # Not qute sure what these are, or what they are for?
        h = np.where(bins_out > 11.2 )
        l = np.where(bins_out <= 11.2 )

        hod = np.zeros_like(bins_out)

        if not Centrals: # Not sure why we are doing this?
            hod[t] = 0.0001
            hod = hist_subject/hist_Centrals_unweighted
        else:
            hod[h] = hist_subject[h]/hist_Centrals_unweighted[h]
            hod[l] = hist_subject[l]/hist_all[l]

        self.HOD_plottingData.append(PlottingData(bins[0:-1], hod))

    def listAttributes(self):
        attributes = [attr for attr in dir(self)
                      if not callable(getattr(self, attr))
                      and not attr.startswith("__")
                      and len(attr) > 1]
        for attr in attributes:
            print(attr, type(attr))

        print(attributes)


def Aird_edd_dist(edd, z):
    prob = np.ones((len(edd)), dtype='float64')

    gammaE = -0.65
    gammaz = 3.47
    z0 = 0.6
    A = 10.**(-3.15)
    prob = A*((edd/(10.**(-1.)))**gammaE)*((1.+z)/(1.+z0))**gammaz
    return prob

def NHfunc(lgLx, z, lgNH):
    '''Funtion that returns the 'absorbsion' function.

    Nh function, as defined by Ueda 2014 (section 3, page 8).

    Attributes:
        lgLx (float) : Log x-ray luminosity.
        z (float) : Redshift
        lgNH (float,) : The 'bins' of Nh
    '''

    ximin, ximax, xi0_4375, a1, eps, fctk, beta = 0.2, 0.84, 0.43, 0.48, 1.7, 1., 0.24

    if z > 2:
        z = 2

    xi_4375 =  (xi0_4375*(1+z)**a1) - beta*(lgLx - 43.75)

    max_ = max(xi_4375, ximin)
    xi = min(ximax, max_)

    fra = (1 + eps)/(3 + eps)

    f = np.ones(len(lgNH))
    flag = np.zeros(len(lgNH))
    flag1 = np.zeros(len(lgNH))
    flag2 = np.zeros(len(lgNH))
    flag3 = np.zeros(len(lgNH))
    flag4 = np.zeros(len(lgNH))

    # Boolean flags to separate conditions
    flag = np.where((lgNH<21) & (lgNH>=20))
    flag1 = np.where((lgNH<22) & (lgNH>=21))
    flag2 = np.where((lgNH<23) & (lgNH>=22))
    flag3 = np.where((lgNH<24) & (lgNH>=23))
    flag4 = np.where((lgNH<26) & (lgNH>=24))

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

    return f; # The value of the function f itself.

class data:
    # Parent class for data type objects
    def __init__(self, z):
        self.dataPath = "./Data/"
        self.z = z
        self.cosmology = cosmology.setCosmology('planck18')
        self.h = self.cosmology.H0/100

class WP_Data(data):

    def __init__(self, z):
        data.__init__(self, z)

        self.r_Koutoulidis = np.array([0.2, 0.35, 0.55, 1, 1.5, 2.5, 3.4, 7.0, 20])
        self.wp_Koutoulidis = np.array([11, 175, 47, 45, 37, 25, 25, 11, 6.1])
        self.wp_Koutoulidis_e = np.array([[50, 100, 20, 10, 10, 1, 1, 1, 1],
                                          [50, 100, 20, 10, 10, 1, 1, 1, 1]])
    def K_powerLaw(self, rp, r0 = 6.2, gam = 1.88):
        rp = rp/self.h
        wp = rp * ((r0/rp) ** gam) * ( (special.gamma(0.5) * special.gamma((gam-1)/2)) / special.gamma(gam/2) )
        return wp *self.h

class XLF_Data(data):

    def __init__(self, z):
        data.__init__(self, z)

        # Read in Miyanji
        #file = self.dataPath + GetCorrectFile("Miyaji2015", self.z, directory = self.dataPath)
        #df = pd.read_csv(file, header = None)
        self.Mi_LX, self.Mi_phi = ReadSimpleFile("Miyaji2015", self.z, self.dataPath)
        self.Mi_phi = np.log10(self.Mi_phi)

    def getMiyaji2015(self):
        return self.Mi_phi, self.Mi_LX

    def getUeda14(self, bins, Nh = 'free' ):
        """
            LF in the 2-10 KeV range based on the article from Ueda et al. (2014) in
            this version of the program we also take into account the detailed
            distribution of the Nh column density with L and z and also the possible
            contribution by Compton-thick AGNs.
        """
        L = bins
        z = self.z

        # Constants
        A = 2.91e-6
        L_s = 43.97
        L_p = 44.
        g_1 = 0.96
        g_2 = 2.71
        p_1s = 4.78
        p_2 = -1.5
        p_3 = -6.2
        b_1 = 0.84
        z_sc1 = 1.86
        L_a1 = 44.61
        alpha_1 = 0.29
        z_sc2 = 3.0
        L_a2 = 44.
        alpha_2 = -0.1

        # Preallocate Memory
        nl = len(L)
        z_c1 = np.zeros(nl)
        z_c2 = np.zeros(nl)
        p_1 = np.zeros(nl)
        e = np.zeros(nl)
        # L_x
        L_x = L.copy()

        z_c1[L_x <= L_a1] = z_sc1 * (10.**(L_x[L_x < L_a1] - L_a1))**alpha_1
        z_c1[L_x > L_a1] = z_sc1
        z_c2[L_x <= L_a2] = z_sc2 * (10.**(L_x[L_x <= L_a2]-L_a2))**alpha_2
        z_c2[L_x > L_a2] = z_sc2

        p_1 = p_1s + b_1*(L_x - L_p)

        e[z <= z_c1] = (1 + z)**p_1[z <= z_c1]
        e[(z > z_c1) & (z < z_c2)] = (1. + z_c1[(z > z_c1) & (z < z_c2)])**p_1[(z > z_c1) & (z < z_c2)] * ((1.+z)/(1.+z_c1[(z > z_c1) & (z < z_c2)]))**p_2
        e[z > z_c2] = (1. + z_c1[z > z_c2])**p_1[z > z_c2] * ((1.+z_c2[z > z_c2])/(1.+z_c1[z > z_c2]))**p_2 * ((1. + z)/(1. + z_c2[z > z_c2]))**p_3

        Den1 = (10.**(L_x - L_s))**g_1
        Den2 = (10.**(L_x - L_s))**g_2
        Den = Den1 + Den2

        Phi = (A/Den) * e

        Psi = np.zeros(nl)
        bet = 0.24
        a1 = 0.48
        fCTK = 1.0
        Psi0 = 0.43


        if z < 2:
            Psi44z = Psi0 * (1.0 + z)**a1
        else:
            Psi44z = Psi0 * (1.+2.)**a1

        eta = 1.7
        Psimax = (1. + eta)/(3. + eta)
        Psimin = 0.2

        em = Psi44z - bet*(L - 43.75)

        Psi = np.ones(nl) * Psimin
        Psi[em > Psimin] = em[em > Psimin]
        Psi[Psimax > Psi] = Psimax

        if Nh == 'free':
            frac = 1.
            return np.log10(Phi), bins

        lim = (1. + eta)/(3. + eta)
        frac = np.zeros(nl)

        for k in range(nl):
         if Psi[k] < lim:
          if Nh == 0:
              frac[k] = 1. - (2. + eta)/(1. + eta)*Psi[k]   #20.<LogNh<21
          if Nh == 1:
              frac[k] = (1./(1. + eta))*Psi[k]          #21.<LogNh<22.
          if Nh == 2:
              frac[k] = (1./(1. + eta))*Psi[k] 			#22.<LogNh<23.
          if Nh == 3:
              frac[k] = (eta/(1.+eta))*Psi[k] 	    #23.<LogNh<24.
          if Nh == 4:
              frac[k] = (fCTK/2.)*Psi[k] 				#24.<LogNh<26
         else:
          if Nh == 0:
              frac[k] = 2./3. - (3. + 2.*eta)/(3. + 3.*eta)*Psi[k]  #20.<LogNh<21
          if Nh == 1:
              frac[k] = 1./3. - eta/(3. + 3.*eta)*Psi[k]          #21.<LogNh<22.
          if Nh == 2:
              frac[k] = (1./(1. + eta))*Psi[k] 					#22.<LogNh<23.
          if Nh == 3:
              frac[k] = (eta/(1. + eta))*Psi[k] 			    #23.<LogNh<24.
          if Nh == 4:
            frac[k] = (fCTK/2.)*Psi[k] 						#24.<LogNh<26

        Phi=frac*Phi
        return np.log10(Phi), bins

class EddingtonDistributionData(data):

    def __init__(self, z):
        data.__init__(self, z)

        # Read in Geo17
        self.Geo_LX, self.Geo_phi = ReadSimpleFile("Geo17", self.z, self.dataPath)

        # Read in Bon 16
        self.Bon16_LX, self.Bon16_phi = ReadSimpleFile("Bon16", self.z, self.dataPath)

        # Read in Bon 12
        self.Bon12_LX, self.Bon12_phi = ReadSimpleFile("Bon12", self.z, self.dataPath)

        # Aird
        self.Aird_LX, self.Aird_phi = ReadSimpleFile("Aird2018", self.z, self.dataPath)

    def AirdDist(self, edd):
        gammaE = -0.65
        gammaz = 3.47
        z0 = 0.6
        A = 10.**(-3.15)
        prob = A*((10**edd/(10.**(-1.)))**gammaE)*((1.+self.z)/(1.+z0))**gammaz

        binwidth = edd[1] - edd[0]
        prob /= binwidth # divide by binwidth
        prob = np.log10(prob[prob > 0])
        return prob


if __name__ == "__main__":
    default = EuclidMaster()
    default.set_z(0.1)
    default.generate_semi_analytic_halos(volume=1000 ** 3)
    default.assign_stellar_mass()
    default.assign_black_hole_mass()
    default.assign_duty_cycle()
    default.assign_luminosity()
    """
    default.assignDutyCycle(0.1)
    default.assignEddingtonRatios()
    default.Obscuration()
    default.listAttributes()
    """
