# Libraries

# Specific Libraries

# Local
import AGNCatalogToolbox as act
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
              ("parent_halo_mass", np.float32),
              ("effective_z", np.float32),
              ("stellar_mass", np.float32),
              ("black_hole_mass", np.float32),
              ("duty_cycle", np.float32),
              ("luminosity", np.float32),
              ("nh", np.float32),
              ("type", np.float32)]
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
            act.generate_semi_analytic_halo_catalogue(catalogue_volume=volume,
                                                      mass_params=(mass_low, mass_high, 0.1),
                                                      z=self.z,
                                                      h=self.h,
                                                      visual_debugging=False,
                                                      erase_debugging_folder=True,
                                                      visual_debugging_path="./visualValidation/SemiAnalyticCatalog/")
        self.define_main_catalog(len(temporary_halos))
        self.main_catalog['effective_halo_mass'] = temporary_halos
        self.main_catalog['parent_halo_mass'] = temporary_halos # Not sure if this is legitimate.

    def assign_stellar_mass(self, formula="Grylls18", scatter=0.001):
        """ Function to generate stellar masses from halo masses

        Just calls the 'class free' function from galaxy_physics

        :param formula: string, the method to use. Options currently include "Grylls18" and "Moster"
        :param scatter: float, the magnitude of the scatter (in dex).
        :return: None
        """
        print("Assigning Stellar Mass")
        self.main_catalog['stellar_mass'] = act.halo_mass_to_stellar_mass(self.main_catalog['effective_halo_mass'],
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
        self.main_catalog['black_hole_mass'] = act.stellar_mass_to_black_hole_mass(self.main_catalog['stellar_mass'],
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
        self.main_catalog["duty_cycle"] = act.to_duty_cycle(function,
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

    def get_wp(self, threads ="System", pi_max = 50, binStart = -1, binStop = 1.5, binSteps = 50):
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
                                    bins=(-1, 1.5, 50),
                                    pi_max=50,
                                    threads="system")

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
        if isinstance(variable, str):
            variable = self.mawin_catalog[variable]

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
        if weight_by_duty_cycle:
            weight = self.main_catalog["duty_cycle"]
        else:
            weight = None
        hod_data = act.calculate_hod(self.main_catalog["up_id"],
                                     self.main_catalog["halo_mass"],
                                     weight,
                                     centrals=centrals)

        self.HOD_plottingData.append(hod_data)


if __name__ == "__main__":
    default = EuclidMaster()
    default.set_z(0.1)
    default.generate_semi_analytic_halos(volume=1000 ** 3)
    default.assign_stellar_mass()
    default.assign_black_hole_mass()
    default.assign_duty_cycle()
    default.assign_luminosity()
    default.assign_obscuration()
    default.get_bias()

