# Libraries

# Specific Libraries

# Local
import AGNCatalogToolbox as act
from ACTUtillity import *


class AGNCatalog:
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
              ("parent_halo_mass", np.float32),
              ("virial_mass", np.float32),
              ("up_id", np.float32),
              ("effective_z", np.float32),
              ("stellar_mass", np.float32),
              ("black_hole_mass", np.float32),
              ("duty_cycle", np.float32),
              ("luminosity", np.float32),
              ("nh", np.float32),
              ("type", np.float32)]
        self.main_catalog = np.zeros(length, dtype=dt)
        self.main_catalog['effective_z'] = np.ones(length) * self.z

    def load_dm_catalog(self, volume_axis=1000, visual_debugging=False, filename="MD_"):
        """ Function to load in the catalog_data from the multi-dark halo catalogue

        This catalog_data should exist as .npy files in the Directory/BigData. Within
        this folder there should be a script to pull these out of the SQL
        database. Note that this expects a .npy file in the with columns x, y, z
        scale factor at accretion, mass at accretion. If generateFigures is set
        to True (default), then a further column of the halo mass is also
        required to validate the halos.

        :param volume_axis: float, length of one axis of the cosmological box, in units of h^-1
        :param visual_debugging: bool, flag to generate the figures that exist for visual validation. Defaults to true.
        :param filename: string, component of the filename excluding the redshift - the closest z will be found
        automatically. Default is "MD_", expecting files of the form "MD_0.0.npy".
        """
        volume_axis = volume_axis/self.h
        self.volume = volume_axis**3  # MultiDark Box size, Mpc

        print("Loading Dark Matter Catalog")
        effective_halo_mass, effective_z, virial_mass, up_id, x, y, z =\
            act.load_halo_catalog(self.h, self.z, self.cosmology,
                                  filename=filename,
                                  path_big_data="./BigData/",
                                  visual_debugging=visual_debugging,
                                  erase_debugging_folder=True,
                                  visual_debugging_path="./visualValidation/NBodyCatalog/")
        self.define_main_catalog(len(effective_halo_mass))

        self.main_catalog["effective_halo_mass"] = effective_halo_mass
        self.main_catalog["effective_z"] = effective_z
        self.main_catalog["virial_mass"] = virial_mass
        self.main_catalog["up_id"] = up_id
        self.main_catalog["x"] = x
        self.main_catalog["y"] = y
        self.main_catalog["z"] = z

    def generate_semi_analytic_halos(self, volume=500**3, mass_low=12., mass_high=16., visual_debugging=False):
        """Function to generate a catalog of semi-analytic halos.

        Function to pull a catalogue of halos from the halo mass function. A
        reasonable volume should be chosen - a larger volume will of course
        produce a greater number of halos, which will increase resolution at
        additional computational expense.

        :param volume: float, The volume of the region within which we will create the halos.
        :param mass_low: float, The lowest mass (in log10 M_sun) halo to generate defaulting to 11.
        :param mass_high: float, The highest mass halo to generate, defaulting to 15
        :param visual_debugging: bool, flag to generate the figures that exist for visual validation. Defaults to true.
        :return: None
        """
        print("Generating Semi-Analytic halos")

        self.dm_type = "Analytic"
        self.volume = volume

        temporary_halos = \
            act.generate_semi_analytic_halo_catalogue(catalogue_volume=volume,
                                                      mass_params=(mass_low, mass_high, 0.1),
                                                      z=self.z,
                                                      h=self.h,
                                                      visual_debugging=visual_debugging,
                                                      erase_debugging_folder=True,
                                                      visual_debugging_path="./visualValidation/SemiAnalyticCatalog/")
        self.define_main_catalog(len(temporary_halos))
        self.main_catalog["effective_halo_mass"] = temporary_halos
        self.main_catalog["parent_halo_mass"] = temporary_halos  # Not sure if this is legitimate.

    def assign_stellar_mass(self, formula="Grylls19", scatter=0.001):
        """ Function to generate stellar masses from halo masses

        Just calls the 'class free' function from galaxy_physics

        :param formula: string, the method to use. Options currently include "Grylls19" and "Moster"
        :param scatter: float, the magnitude of the scatter (in dex).
        :return: None
        """
        print("Assigning Stellar Mass")
        self.main_catalog["stellar_mass"] = act.halo_mass_to_stellar_mass(self.main_catalog["effective_halo_mass"],
                                                                          self.main_catalog["effective_z"],
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
        :param scatter: string or float, string should be "Intrinsic", float value specifies the (fixed) scatter
        magnitude
        :return: None
        """
        print("Assigning Black Hole Mass")
        self.main_catalog['black_hole_mass'] = act.stellar_mass_to_black_hole_mass(self.main_catalog['stellar_mass'],
                                                                                   method=formula,
                                                                                   scatter=scatter,
                                                                                   visual_debugging=False,
                                                                                   erase_debugging_folder=True,
                                                                                   debugging_volume=self.volume,
                                                                                   visual_debugging_path=
                                                                                   "./visualValidation/BlackHoleMass/")

    def assign_duty_cycle(self, function="Man16"):
        """Function to assign black hole masses atop stellar masses.

        :param function: string/float, string specifying the method, options are "Man16"/"Schulze", or a constant float
        :return: None
        """
        print("Assigning Duty Cycle, using {}'s method".format(function))
        self.main_catalog["duty_cycle"] = act.to_duty_cycle(function,
                                                            self.main_catalog['stellar_mass'],
                                                            self.main_catalog['black_hole_mass'],
                                                            self.z)

    def assign_luminosity(self, method="Schechter", redshift_evolution=False, parameter1=-1, parameter2=-0.65):
        """ Function to assign luminosity (based on black hole mass)

        :param method: string, the function to pull the Eddington Ratios from. Options are "Schechter", "PowerLaw" or
        "Gaussian".
        :param redshift_evolution: bool, if set to true will introduce a factor representing the z-evolution.
        :param parameter1: float, the first parameter for the method. For Schechter it is the knee, for PowerLaw it is
        not used, and for the Gaussian it is sigma.
        :param parameter2: float, the second parameter for the method. For Schechter it is alpha, for PowerLaw it is
        not used, for Gaussian it is b.
        :return: None
        """
        print("Assigning Luminosity")
        self.main_catalog['luminosity'], xlf_plotting_data, edd_plotting_data = \
            act.black_hole_mass_to_luminosity(self.main_catalog["black_hole_mass"],
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

    def assign_obscuration(self):
        """ Function to assign both nh and the AGN type (type 1, type 2 etc). These are rolled together for convenience.

        :return: None
        """
        print("Assigning Nh")
        self.main_catalog["nh"] = act.luminosity_to_nh(self.main_catalog["luminosity"], self.z)
        print("Assigning AGN type")
        self.main_catalog["type"] = act.nh_to_type(self.main_catalog["nh"])

    def get_wp(self, threads="System", pi_max=50, bins=(-1, 1.5, 50)):
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
        wp_results = act.compute_wp(self.main_catalog["x"],
                                    self.main_catalog["y"],
                                    self.main_catalog["z"],
                                    period=self.volume**(1/3),
                                    weights=self.main_catalog["duty_cycle"],
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

        bias_result = act.compute_bias(variable, self.main_catalog['parent_halo_mass'],
                                       self.z, self.h, self.cosmology, bin_size=bin_size, weight=weight, mask=mask)
        self.bias_plottingData.append(bias_result)

    def get_hod(self, centrals=True, weight_by_duty_cycle=True):
        """ Fuction to calculate the HOD of the catalog

        :param centrals: bool, if we should only calculate the HOD for centrals
        :param weight_by_duty_cycle: bool, if we should weight by duty cycle.
        :return: None.
        """
        print("Calculating HOD")
        if weight_by_duty_cycle:
            weight = self.main_catalog["duty_cycle"]
        else:
            weight = None
        hod_data = act.calculate_hod(self.main_catalog["up_id"],
                                     self.main_catalog["virial_mass"],
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

