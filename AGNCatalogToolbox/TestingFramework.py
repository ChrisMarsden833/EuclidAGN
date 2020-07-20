# Libraries
import numpy as np

# Specific Libraries
from colossus.cosmology import cosmology

# Local
from AGNCatalogToolbox import main
from AGNCatalogToolbox import Utillity as utl


class AGNCatalog:
    """Class that encapsulates the AGN toolbox code.

    This object should be created, and then it's member functions should be
    called in the appropriate order with varying parameters where desired. The
    'getter' functions can then be used to extract relevant data.

    Constructor parameters
    :param z: float, the redshift of the catalog
    :param cosmo: string, describing the cosmology the class will use. The default is planck18.
        This is passed straight to colossus.cosmology, so define as appropriate.
    """
    def __init__(self, z=0, cosmo='planck18', print_updates=True):
        # Setup
        self.z = z
        self.cosmology_name = cosmo # Store this as it's useful
        self.cosmology = cosmology.setCosmology(cosmo)

        # Extract Reduced Hubble Constant, because we use it so much
        self.h = self.cosmology.h

        # Image and data file locations - check they exist
        self.path_data = utl.ValidatePath("./Data/", ErrorOnFail=True)
        self.path_bigData = utl.ValidatePath("./BigData/", ErrorOnFail=True)

        # Initialization (not strictly necessary but for PEP8 compliance...
        self.dm_type = 'Undefined'
        self.volume_axis = 0
        self.volume = 0

        self.wpbins = None

        # Primary Catalog
        self.main_catalog = []

        # PlottingData. We don't use this at the moment.
        '''
        self.XLF_plottingData = []
        self.Edd_plottingData = []
        self.WP_plottingData = []
        self.bias_plottingData = []
        self.HOD_plottingData = []
        '''

        self.process_bookkeeping = {
            "Dark Matter": False,
            "Stellar Mass": False,
            "Black Hole Mass": False,
            "Eddington Ratios": False,
            "Duty Cycle": False}

        self.DC_name = None

    def define_main_catalog(self, length):
        """ Internal function to define the main catalog

        :param length: the size of the data to reserve.
        :return: None
        """
        dt = [("x", np.float32),
              ("y", np.float32),
              ("z", np.float32),
              ("effective_halo_mass", np.float32),
              ("parent_halo_mass", np.float32),
              ("virial_mass", np.float32),
              ("up_id", np.float32),
              ("effective_z", np.float32),
              ("stellar_mass", np.float32),
              ("black_hole_mass", np.float32),
              ("duty_cycle", np.float32),
              ("Eddington_Ratio", np.float32),
              ("luminosity", np.float32),
              ("nh", np.float32),
              ("type", np.float32)]
        self.main_catalog = np.zeros(length, dtype=dt)
        self.main_catalog['effective_z'] = np.ones(length) * self.z

    def load_dm_catalog(self, volume_axis=1000, filename="MD_", path_big_data="./BigData/"):
        """ Function to load in the catalog_data from the multi-dark halo catalogue

        This catalog_data should exist as .npy files in the Directory/BigData. Within
        this folder there should be a script to pull these out of the SQL
        database. Note that this expects a .npy file in the with columns x, y, z
        scale factor at accretion, mass at accretion. If generateFigures is set
        to True (default), then a further column of the halo mass is also
        required to validate the halos.

        :param volume_axis: float, length of one axis of the cosmological box, in h^-1
        :param filename: string, component of the filename excluding the redshift - the closest z will be found
        automatically. Default is "MD_", expecting files of the form "MD_0.0.npy".
        :return None
        """
        volume_axis = volume_axis/self.h
        self.volume = volume_axis**3  # MultiDark Box size, Mpc

        print("Loading Dark Matter Catalog")
        effective_halo_mass, effective_z, virial_mass, up_id, x, y, z =\
            main.load_halo_catalog("auto", self.z, self.h, filename_prefix=filename, path_big_data=path_big_data)
        self.define_main_catalog(len(effective_halo_mass))
        self.main_catalog["effective_halo_mass"] = effective_halo_mass
        self.main_catalog["effective_z"] = effective_z
        self.main_catalog["virial_mass"] = virial_mass
        self.main_catalog["up_id"] = up_id
        self.main_catalog["x"] = x
        self.main_catalog["y"] = y
        self.main_catalog["z"] = z

    def generate_semi_analytic_halos(self, volume=500**3, mass_tuple=(12, 16, 0.1)):
        """Function to generate a catalog of semi-analytic halos.

        Function to pull a catalogue of halos from the halo mass function. A
        reasonable volume should be chosen - a larger volume will of course
        produce a greater number of halos, which will increase resolution at
        additional computational expense.

        :param volume: float, The volume of the region within which we will create the halos.
        :param mass_tuple, tuple of floats (mass low, mass high, spacing) representing the domain over which masses
            should be created [log10 M_sun].
        :param visual_debugging: bool, flag to generate the figures that exist for visual validation. Defaults to true.
        :return: None
        """
        print("Generating Semi-Analytic halos")

        self.dm_type = "Analytic"
        self.volume = volume

        temporary_halos = \
            main.generate_semi_analytic_halo_catalogue(catalogue_volume=volume, mass_params=mass_tuple, z=self.z,
                                                       h=self.h)
        self.define_main_catalog(len(temporary_halos))
        self.main_catalog["effective_halo_mass"] = temporary_halos
        self.main_catalog["parent_halo_mass"] = temporary_halos  # Not sure if this is legitimate.

    def get_hmf(self, bins="infer", return_colossus=False):
        """Wrapper function to return the (plotting data for the) halo mass function for the catalog.

        :param bins: (numpy) array, bins for the hmf. Default is the only exeption, a string "infer" which will result
            an array with the highest and lowest values in the halo mass catalog, with a spacing of 0.1 [log10 M_sun].
        :param return_colossus: bool, if the hmf from colossus should also be returned.
        :return: None
        """
        if not isinstance(return_colossus, (bool)):
            raise ValueError("compare should be a bool, got: {}".format(return_colossus))

        if bins is "infer":
            default_spacing = 0.1
            bins = np.arange(np.amin(self.main_catalog["effective_halo_mass"]),
                             np.amax(self.main_catalog["effective_halo_mass"]),
                             default_spacing)
        if not isinstance(bins, (list, tuple, np.ndarray)):
            raise ValueError("Bins should be an array of numeric values, got {}".format(bins))

        if not return_colossus:
            hmf, bins = main.halo_mass_function(self.main_catalog["effective_halo_mass"], bins, self.volume)
            return hmf, bins
        if return_colossus:
            hmf, bins, chmf, cbins = main.halo_mass_function(self.main_catalog["effective_halo_mass"], bins, self.volume,
                                                      compare=return_colossus, z=self.z, h=self.h)
            return hmf, bins, chmf, cbins

    def assign_stellar_mass(self, formula="Grylls19", scatter=0.11):
        """ Function to generate stellar masses from halo masses

        Just calls the 'class free' function from galaxy_physics

        :param formula: string, the method to use. Options currently include "Grylls19" and "Moster"
        :param scatter: float, the magnitude of the scatter [log10 M_sun]
        :return: None
        """
        self.main_catalog["stellar_mass"] = main.halo_mass_to_stellar_mass(self.main_catalog["effective_halo_mass"],
                                                                           self.main_catalog["effective_z"],
                                                                           formula=formula,
                                                                           scatter=scatter)

    def assign_black_hole_mass(self, formula="Shankar16", scatter="Intrinsic"):
        """Function to generate black hole masses atop stellar masses.

        :param formula: string, specifying the method to be used, options are "Shankar16",  "KormondyHo" and "Eq4".
        :param scatter: string or float, string should be "Intrinsic", float value specifies the (fixed) scatter
        magnitude. Float values are in [log10 M_sun]
        :return: None
        """
        self.main_catalog['black_hole_mass'] = main.stellar_mass_to_black_hole_mass(self.main_catalog['stellar_mass'],
                                                                                   method=formula, scatter=scatter)

    def assign_duty_cycle(self, function="Man16"):
        """Function to assign black hole masses atop stellar masses.

        :param function: string/float, string specifying the method, options are "Man16"/"Schulze", or a constant float
        :return: None
        """
        self.main_catalog["duty_cycle"] = main.to_duty_cycle(function,
                                                            self.main_catalog['stellar_mass'],
                                                            self.main_catalog['black_hole_mass'],
                                                            self.z)
        self.DC_name = function

    def assign_luminosity_eddington_ratio(self, method="Schechter", parameter1=-1, parameter2=-0.65,
                                          redshift_evolution=False):
        """ Function to assign luminosity (based on black hole mass)

        :param method: string, the function to pull the Eddington Ratios from. Options are "Schechter", "PowerLaw" or
        "Gaussian".
        :param parameter1: float, the first parameter for the method. For Schechter it is the knee, for PowerLaw it is
        not used, and for the Gaussian it is sigma.
        :param parameter2: float, the second parameter for the method. For Schechter it is alpha, for PowerLaw it is
        not used, for Gaussian it is b.
        :param redshift_evolution: bool, if set to true will introduce a factor representing the z-evolution.
        :return: None
        """

        self.main_catalog["luminosity"], self.main_catalog["Eddington_Ratio"] =\
            main.bh_mass_to_eddington_ratio_luminosity(self.main_catalog["black_hole_mass"], self.z, method=method,
                                                       redshift_evolution=redshift_evolution, parameter1=parameter1,
                                                       parameter2=parameter2)


    def assign_nh(self, parallel=True):
        """ Function to assign nh
        :param parallel: boolean, run the code in parallel
        :return: None
        """
        self.main_catalog["nh"] = main.luminosity_to_nh(self.main_catalog["luminosity"], self.z, parallel=parallel)

    def assign_type(self):
        print("Assigning AGN type")
        self.main_catalog["type"] = main.nh_to_type(self.main_catalog["nh"])

    def get_function(self, name, bins, weight=True):
        """ Function to generate the statistical function of a property based on it's name, over domain of bins.

        :param name: string, the name of the variable that we want to create the function for.
        :param bins: (numpy) array of bins that we want to bin our variable by [appropriate units]
        :param weight: bool, if we should weight by the duty cycle.
        :return: (numpy) array, the function itself [appropriate units], (numpy) array, the bins are also returned.
        """
        if weight:
            return main.weightedFunction(self.main_catalog[name], self.main_catalog["duty_cycle"], bins, self.volume), \
                   bins[:-1]
        else:
            weights = np.ones_like(self.main_catalog["duty_cycle"])
            return main.weightedFunction(self.main_catalog[name], weights, bins, self.volume), bins[:-1]




    def get_wp(self, threads="System", pi_max=50, bins=(-1, 1.5, 50), cut=True):
        """Function to compute the correlation function wp

        Attributes:
            threads (int/string) : The specifier for the number of threads to create.
                default is "System", which uses all available cores.
            pi_max (float) : The value of pi max for wp. Defaults to 50
            binStart (float) : Low limit to the (log spaced) bins. Defaults to 1
            binStop (float) : High limit to the (log spaced) bins. Defaults t0 1.5
            binSteps (int) : The number of spaces in the bins. Defaults to 50
        """
        if cut:
            '''
            flag = (self.main_catalog["luminosity"] >= 42.01) *\
                   (self.main_catalog["luminosity"] <= 44.98) *\
                   (self.main_catalog["type"] >= 2)
            '''
            flag = (self.main_catalog["luminosity"] >= 42.01) *\
                   (self.main_catalog["luminosity"] <= 44.98) *\
                   (self.main_catalog["nh"] > 22)
        else:
            flag = np.ones_like(self.main_catalog["x"])

        wp_results = main.compute_wp(self.main_catalog["x"][flag],
                                    self.main_catalog["y"][flag],
                                    self.main_catalog["z"][flag],
                                    period=(self.volume**(1/3))*self.h,
                                    weights=self.main_catalog["duty_cycle"][flag],
                                    bins=bins,
                                    pimax=pi_max,
                                    threads=threads)
        self.wpbins=np.logspace(bins[0], bins[1], bins[2])
        self.WP_plottingData.append(wp_results)

    def get_bias(self, variable="stellar_mass", bin_size=0.3, weight="duty_cycle", mask=None):
        """ Function to compute the bias against a specified variable.

        :param variable: string/array, the variable to test against. If a string, will try to extract the array with
        this name from the main_catalogue.
        :param bin_size: float, width of the bins.
        :param weight: string/array. Weights. If a string, will try to extract the array with this name from the
        main_catalogue (normally 'duty_cycle').
        :param mask: array/None. If an array, mask the bias calculation.
        :return: None.
        """
        print("Calculating Bias")
        if isinstance(variable, str):
            variable = self.main_catalog[variable]

        if isinstance(weight, str):
            weight = self.main_catalog[weight]

        bias_result = main.compute_bias(variable, self.main_catalog['parent_halo_mass'],
                                       self.z, self.h, self.cosmology, bin_size=bin_size, weight=weight, mask=mask)
        self.bias_plottingData.append(bias_result)

    def get_hod(self, centrals=True, weight_by_duty_cycle=True, Obscuration="Both"):
        """ Fuction to calculate the HOD of the catalog

        :param centrals: bool, if we should only calculate the HOD for centrals
        :param weight_by_duty_cycle: bool, if we should weight by duty cycle.
        :return: None.
        """
        print("Calculating HOD")

        if Obscuration=="Both":
            flag = np.ones_like(self.main_catalog["up_id"])
        elif Obscuration=="Obscured":
            flag = (self.main_catalog["nh"] > 22)
        elif Obscuration=="Unobscured":
            flag = (self.main_catalog["nh"] < 22)
        else:
            assert False, "Unknown obscuration type {}".format(Obscuration)


        if weight_by_duty_cycle:
            weight = self.main_catalog["duty_cycle"][flag]
        else:
            weight = None

        hod_data = main.calculate_hod(self.main_catalog["up_id"][flag],
                                     self.main_catalog["virial_mass"][flag],
                                     weight,
                                     centrals=centrals)

        self.HOD_plottingData.append(hod_data)


if __name__ == "__main__":
    default = AGNCatalog()
    default.set_z(0.1)
    default.generate_semi_analytic_halos(volume=1000 ** 3)
    default.assign_stellar_mass()
    default.assign_black_hole_mass()
    default.assign_duty_cycle()
    default.assign_luminosity()
    default.assign_obscuration()
    default.get_bias()

