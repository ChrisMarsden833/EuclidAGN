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

# Specific Libraries
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import Corrfunc
from Corrfunc.theory import wp
from Corrfunc.theory import DD

class EuclidMaster:
    """Class that encapsulates the Euclid code.

    This object should be created, and then it's member functions should be
    called in the approprate order with varying parameters where desired. The
    'getter' functions can then be used to extract relavant data.

    Attributes:
        cosmo (string): A string describing the cosmology the class will use.
            The default is planck18. This is passed straight to
            colossus.cosmology, so define as approprate.
    """

    def __init__(self, cosmo = 'planck18'):
        self.cosmology = cosmology.setCosmology(cosmo)
        # Extract Reduced Hubble Constant, because we use it so much
        self.h = self.cosmology.H0/100
        # Image and data file location
        self.visualValidatePath = "./visualValidation/"
        self.dataPath = "./Data/"
        self.bigData = "./BigData/"

        self.XLF_plottingData = []
        self.Edd_plottingData = []
        self.WP_plottingData = []
        self.bias_plottingData = []

    def setRedshift(self, z = 0):
        self.z = z

    def TestForRequiredVariables(self, Names):
        """Function to check that the supplied list of variables actually exist.

        This is for internal use. Will throw an assertion error if names do not
        exist.

        Arguments:
            Names (array of strings) : the names of the variables to search for.
        """
        for name in Names:
            assert hasattr(self, name), "You need to assign {} first".format(name)

    def DarkMatter(self, type = "MultiDark"):
        if type == "MultiDark":
            self.loadMultiDarkHaloes() # TODO put these arguments into a parameter file.
        elif type == "Analytic":
            self.generateSemiAnalyticHaloes()
        else:
            assert False, "Unknown DarkMatter type {}".format(type)

    def loadMultiDarkHaloes(self, generateFigures = True, filename = "MD_" ):
        """ Function to load in the data from the multidark halo catalogue

        This data should exist as .npy files in the Directory/BigData. Within
        this folder there should be a script to pull these out of the SQL
        database. Note that this expects a .npy file in the with columns x, y, z
        scale factor at accretion, mass at accretion. If generateFigures is set
        to True (default), then a further column of the halo mass is also
        required to validate the haloes.

        Attributes:
            generateFigures (bool) : flag to generate the figures that exist for
                visual validation. Defaults to true.
            data_directory (string) : relative path to the directory containing
                the data. Defaults to ./BigData
            filename (string) : component of the filename excluding the redshift
                - the closest z will be found automatically. Default is "MD_",
                expecting files of the form "MD_0.0.npy".
        """
        print("Loading MultiDark Haloes")
        self.halotype = "N-Body" # Flag used for later
        self.axis = 1000/self.h
        self.volume = (self.axis)**3 # MultiDark Box size, Mpc

        # Search for the closest file and read it in.
        file, retz = GetCorrectFile(filename, self.z, directory = self.bigData, retz = True)
        data = np.load(self.bigData + file) # most of this is in h^-1 Mpc

        # If we want to generate the Halo Mass function figure, this section.
        if generateFigures:
            width = 0.1
            bins = np.arange(10, 15, width)
            hist = np.histogram(np.log10(data["mvir"]/self.h), bins = bins)[0]
            volume = self.volume
            hmf = (hist/(volume))/(width)
            fig = plt.figure()
            plt.plot(10**bins[0:-1], hmf, 'o', label = "Multi-Dark")
            try:
                binwidth = 0.01
                M = 10**np.arange(10.0, 15.0, binwidth) + np.log10(self.h) # In unit's of h, for now.
                mfunc = mass_function.massFunction(M*self.h, self.z, mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')*np.log(10) *(self.h**3) #dn/dlog10M
                plt.plot(M, mfunc, label = "Colossus")
            except:
                print("Colossus Failed to plot HMF")
            plt.xlabel("Halo Mass $M_\odot$")
            plt.ylabel(r'$d\phi /d(log\;L_x)\;[Mpc^{-3}]$')
            plt.title("Halo Mass function from Multidark, z = {}".format(self.z))
            plt.loglog()
            plt.legend()
            # Write the file to the approprate location.
            savePath = self.visualValidatePath + 'MultiDark_HMF.png'
            fig.savefig(savePath)
            plt.close()

        # Store the data in class variables, correcting for h.

        x_coord = data['x']/self.h
        y_coord = data['y']/self.h
        z_coord = data['z']/self.h
        id = data["id"]
        upid = data["upid"]
        mvir = data["mvir"]/self.h
        macc = data["Macc"]/self.h
        aacc = data["Acc_Scale"]

        del data # Save on memory

        # Reserved mem
        mvir_par = np.zeros_like(mvir)
        idh = np.zeros_like(mvir)
        effective_z = np.zeros_like(aacc)

        print('    sorting list w.r.t. upId')
        inds = upid.argsort() # + 1 Array, rest are reused
        x_coord = x_coord[inds]
        y_coord = y_coord[inds]
        z_coord = z_coord[inds]
        id = id[inds]
        upid = upid[inds]
        mvir = mvir[inds]
        macc = macc[inds]
        aacc = aacc[inds]
        upid0 = np.searchsorted(upid, 0) # Position where zero should sit on the now sorted upid.
        # All arrays are now sorted by upid

        print('    copying all ', str(upid0), 'elements with upid = -1')
        mvir_par[:upid0] = mvir[:upid0] # MVir of centrals, or where upid  = -1, so just the mass.
        #idh[:upid0] = 0 # Does nothing?

        print('    sorting remaining list list w.r.t. rockstarId')
        upid_cut = upid[upid0:] # Upids that are not -1, or the satalites, with a value pointing to their progenitor.

        id_cut = id[:upid0] # ids of centrals
        mvir_cut = mvir[:upid0] # masses of centrals

        inds = id_cut.argsort() # get indexes to centrals by id.
        id_cut = id_cut[inds] # actually sort centrals by id.
        mvir_cut = mvir_cut[inds] # sort virial masses the same way.

        print('    copying remaining', str(len(upid) - upid0), 'elements')
        inds = np.searchsorted(id_cut, upid_cut) # indexes of where satalite id's point to centrals
        mvir_par[upid0:] = mvir_cut[inds] # Sort mvir_par by this, and assign it to the sat part of mvir_par
        # This gives us the virial mass of the parent or itself if it is a parent. But do we acutally need this?
        idh[upid0:] = 1

        halo_mass = mvir
        halo_mass[idh > 0] = macc[idh > 0]

        effective_z[idh > 0] = 1/aacc[idh > 0] - 1
        effective_z[idh < 1] = retz

        #flagsat = np.where(idh > 0)
        #flagcen = np.where(idh < 1)

        #self.Mvir = data["Mvir"]/self.h
        #self.First_Acc_Mvir = data["First_Acc_Mvir"]/self.h

        self.x_coord = x_coord
        self.y_coord = y_coord
        self.z_coord = z_coord
        self.effective_halo_mass = np.log10(halo_mass)
        self.effective_z = effective_z
        self.parent_halo_mass = np.log10(mvir_par)

    def generateSemiAnalyticHaloes(self, volume = 500, mass_low = 12., mass_high = 16., generateFigures = True):
        """Function to generate a cataloge of Semi-Analyic Haloes.

        Function to pull a catalogue of haloes from the halo mass function. A
        reasonable volume should be chosen - a larger volume will of course
        produce a greater number of haloes, which will increase resolution at
        additional computational expense.

        Attributes:
            volume (float) : The volume of the region within which we will
                create the haloes.
            mass_low (float) : The lowest mass (in log10 M_sun) halo to generate
                defaulting to 11.
            mass_high (float) : The highest mass halo to generate, defaulting to
                15
            generateFigures (bool) : flag to generate the figures that exist for
                visual validation. Defaults to true.
        """
        print("Generating Semi-Analyic Haloes")
        # Preliminary tests
        self.halotype = "Analytic"
        self.volume = volume

        # Create HMF
        binwidth = 0.1 # Well defined binwidth is vital for this process.
        # First we define a possible 'mass range'
        M = 10**np.arange(mass_low, mass_high, binwidth) + np.log10(self.h) # In unit's of h
        # Using it, we then generate a mass function using colossus. Unit corrections kindly suppled by Lorenzo.
        mfunc = mass_function.massFunction(M, self.z, mdef = '200m', model = 'tinker08', q_out = 'dndlnM') * np.log(10)/self.h  # dn/dlog10M

        # Plot the halo mass function generated by colossus.
        if generateFigures:
            fig = plt.figure()
            plt.xlabel('M200m')
            plt.ylabel('M')
            plt.title('Halo Mass function from Colossus')
            plt.loglog()
            plt.plot(M, mfunc, '-', label = 'z = %.1f' % self.z)
            plt.legend()
            savePath = self.visualValidatePath + 'Colossus_HMF.png'
            fig.savefig(savePath)
            plt.close()

        # We determine the Cumulative HMF starting from the high mass end, multipled by the bin width.
        # This effectively gives the cumulative probabiliy of a halo existing.
        cum_mfunc = np.flip(np.cumsum(np.flip(mfunc, 0)), 0) * binwidth

        ########################################################################
        # Interpolation Tests
        # Interpolator for the testing - we will update this with the volume in a second.
        # This is essenially for a volume of size unity.
        interpolator = sp.interpolate.interp1d(cum_mfunc, M)

        sample_index = int(np.floor(len(cum_mfunc)/2)) # index of the half way point in the cumulative function.
        num_test = cum_mfunc[sample_index] # The value of the cum function at this index
        mass_test = interpolator(num_test) # Interpolate to get the mass that this predicts
        # Check that these values are equivilant.
        assert M[sample_index] == mass_test, "Interpolation method incorrect: Back interpolation at midpoint failed"
        # Check first element is equivilant to the total (it obviously must be, but it's worth checking)- to 10 SF accuracy
        accuracy = 10
        assert np.round(cum_mfunc[0], accuracy) == np.round(np.sum(mfunc) * binwidth, accuracy), "Final Cumsum element != total sum"
        ########################################################################

        # Multiply by volume.
        cum_mfunc = cum_mfunc * volume

        # Get the maximum cumulative number.
        max_number = np.floor(cum_mfunc[0])
        range_numbers = np.arange(max_number)

        # Update interpolator
        interpolator = sp.interpolate.interp1d(cum_mfunc, M)
        masses_cataloge = interpolator(range_numbers[range_numbers >= np.amin(cum_mfunc)])

        # Reconstruct HMF
        width = 0.1
        bins = np.arange(10, 16, width)
        hist = np.histogram(np.log10(masses_cataloge), bins=bins)[0]
        hmf = (hist/(volume))/width

        # Plot both as a comparison
        if generateFigures:
            # Plot both as a comparison
            plt.figure()
            plt.loglog()

            bins_power = 10**(bins[0:-1])

            plt.plot(bins_power[hmf != 0], hmf[hmf != 0], 'o', label = 'Reconstructed')

            plt.plot(M, mfunc, label = 'Original');
            plt.legend()
            plt.xlabel("Halo Mass")
            plt.ylabel("Number Density")
            plt.title("Reconstruncted HMF")
            savePath = self.visualValidatePath + 'HMF_Validation.png'
            plt.savefig(savePath)
            plt.close()

        self.effective_halo_mass = np.log10(masses_cataloge)

    def assignStellarMass(self, formula = "Grylls18", scatter = True, scatter_scale = 0.001, generateFigures = True):
        """Function to generate stellar masses from halo masses.

        This is based on Grylls 2018, but also has the option to use the
        parameters from Moster. This is a simplified version of Pip's
        DarkMatterToStellarMass() function.

        Attributes:
            formula (string) : The prescription to use. This defaults to
                "Grylls18", and can also be set to "Moster"
            scatter (bool) : flag to switch on the scatter. Defaults to True.
            scatter_scale (float): the scale of the scatter (not in log10).
                Defaults to 0.001.
            generateFigures (bool) : flag to generate the figures that exist for
                visual validation. Defaults to true.
        """
        print("Assigning Stellar Mass")
        # Tests
        self.TestForRequiredVariables(["effective_halo_mass", "effective_z"])

        if self.halotype == "N-Body":
            working_redshift = self.effective_z # TODO Update for only for satalites - Macc, Scale at acc, Centrals - redshift now, mass now
        else:
            working_redshift = self.z

        # If conditions to set the correct parameters.
        if formula == "Grylls18":
            zparameter = np.divide(working_redshift - 0.1, working_redshift + 1)
            M10, SHMnorm10, beta10, gamma10, Scatter = 11.95, 0.032, 1.61, 0.54, 0.11
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
        elif formula == "Moster":
            zparameter = np.divide(working_redshift, working_redshift + 1)
            M10, SHMnorm10, beta10, gamma10 = 11.590, 0.0351, 1.376, 0.608
            M11, SHMnorm11, beta11, gamma11 = 1.195, -0.0247, -0.826, 0.329

        # Create full parameters
        M = M10 + M11*zparameter
        N = SHMnorm10 + SHMnorm11*zparameter
        b = beta10 + beta11*zparameter
        g = gamma10 + gamma11*zparameter
        # Full formula
        #DM = self.effective_halo_mass
        SM = np.log10(np.power(10, self.effective_halo_mass) * (2*N*np.power( (np.power(np.power(10,self.effective_halo_mass-M), -b) + np.power(np.power(10,self.effective_halo_mass-M), g)), -1)))
        del self.effective_halo_mass
        # Add scatter, if requested.
        if scatter:
            SM += np.random.normal(scale = scatter_scale, size = np.shape(SM))
        # Assign Stellar Mass to the class.
        self.stellar_mass = SM
        # Generate the figures, if requested.
        if generateFigures:
            width = 0.1
            bins = np.arange(9, 15, width)

            hist = np.histogram(self.stellar_mass, bins = bins)[0]
            hmf = (hist/(self.volume))/width
            log_smf = np.log10(hmf[hmf != 0])
            adj_bins = bins[0:-1][hmf != 0]

            plt.figure()
            plt.loglog()
            plt.plot(10**adj_bins, (10**log_smf), label = "Grylls 2019")
            plt.xlabel("Stellar Mass")
            plt.ylabel("phi")
            plt.title("Stellar Mass Function, assigned from Pip's code")
            plt.legend()
            savePath = self.visualValidatePath + 'SMF_Validation.png'
            plt.savefig(savePath)
            plt.close()

    def assignBlackHoleMass(self, SMBH = "Francesco", scatter = "intrinsic", scatter_magnitude = 0.6, generateFigures = True):
        """Function to assign black hole mass from Stellar Mass

        This is based on prescriptions from Shankar and Kormondy and Ho.

        Attributes:
            SMBH (string) : The prescription to use. Defaults to "Francesco"
                which represents the Shankar 2016 prescription. Other options
                include "KormondyHo" and "Eq4".
            scatter (string) : the 'type' of scatter to be use. Defaults to
                "intrinsic", which is the scatter suggested by the prescription.
                Set to "fixed" to introduce a constant scatter.
            scatter_magnitude (float) : The size of the fixed scatter (dex).
                Obviously only meaningful if scatter_magnitude is set to fixed.
                Defaults to 0.6.
            generateFigures (bool) : flag to generate the figures that exist for
                visual validation. Defaults to true.
        """
        print("Assigning Black Hole Mass")
        # Tests
        self.TestForRequiredVariables(["z", "stellar_mass"])

        if SMBH == "Francesco":
            log_Black_Hole_Mass = \
                7.574 + 1.946 * (self.stellar_mass - 11) - \
                0.306 * (self.stellar_mass - 11)**2. - 0.011 * (self.stellar_mass - 11)**3.
            if scatter == "intrinsic":
                log_Black_Hole_Mass += (0.32 - 0.1*(self.stellar_mass - 12.)) * np.random.normal(0., 1.,len(self.stellar_mass))
            elif scatter == "fixed":
                log_Black_Hole_Mass += np.random.normal(0., scatter_magnitude, len(self.stellar_mass))
        elif SMBH == "KormondyHo":
            log_Black_Hole_Mass = 8.54 + 1.18 * (self.stellar_mass - 11)
            if scatter == "intrinsic":
                print("Warning - Kormondy and Ho's intrinsic scatter is effectively fixed, with a scale of 0.5")
                scatter = np.random.normal(0, 0.5, len(self.stellar_mass))
                log_Black_Hole_Mass += scatter
            elif scatter == "fixed":
                scatter = np.random.normal(0, scatter_magnitude, len(self.stellar_mass))
                log_Black_Hole_Mass += scatter
        elif SMBH == 'Eq4':
            log_Black_Hole_Mass =  8.35 + 1.31 * (Stellar_Mass - 11)
            if scatter == "intrinsic":
                print("Warning - Eq4's intrinsic scatter is effectively fixed, with a scale of 0.5")
                scatter = np.random.normal(0, 0.5, len(self.stellar_mass))
                log_Black_Hole_Mass += scatter
            elif scatter == "fixed":
                scatter = np.random.normal(0, scatter_magnitude, len(self.stellar_mass))
                log_Black_Hole_Mass += scatter
        else:
            assert False, "Unknown SMBH type - {}".format(SMBH)

        self.SMBH_mass = log_Black_Hole_Mass

        if generateFigures:
            width = 0.1
            bins = np.arange(6, 10, width)

            hist = np.histogram(self.SMBH_mass, bins = bins)[0]
            bhmf = (hist/(self.volume))/width
            log_bhmf = np.log10(bhmf[bhmf != 0])
            adj_bins = bins[0:-1][bhmf != 0]

            plt.figure()
            plt.loglog()
            plt.plot(10**adj_bins, (10**log_bhmf), label = "{}".format(SMBH))
            plt.xlabel("Black Hole Mass")
            plt.ylabel("phi")
            plt.title("Black Hole Mass Function")
            plt.legend()
            savePath = self.visualValidatePath + 'BHMF_Validation.png'
            plt.savefig(savePath)
            plt.close()

    def assignDutyCycle(self, function = "Mann"):
        print("Assigning Duty Cycle, {}".format(function))
        self.TestForRequiredVariables(["SMBH_mass", "stellar_mass"])

        type_cost = type(function)
        if type_cost is float or type_cost is int:
            DC = np.ones_like(self.SMBH_mass) * function
            self.dutycycle = DC
        elif isinstance(function, str):
            if function == "Mann":
                if self.z > 0.1:
                    print("Warning - Mann's duty cycle is not set up for redshifts other than zero")
                Mann_path = self.dataPath + "Mann.csv"
                df = pd.read_csv(Mann_path, header = None)
                SM = df[0]
                U = df[1]
                fn = sp.interpolate.interp1d(SM, U)
                output = np.zeros_like(self.stellar_mass)
                output[self.stellar_mass < np.amin(SM)] = np.amin(U)
                output[self.stellar_mass > np.amax(SM)] = np.amax(U)
                cut = (self.stellar_mass < np.amax(SM)) * (self.stellar_mass > np.amin(SM))
                output[cut] = fn(self.stellar_mass[cut])
                self.dutycycle = 10**output

            elif function == "Schulze":
                # Find the nearest file to the redshift we want
                schulzePath = self.dataPath + GetCorrectFile("Schulze", self.z, self.dataPath)
                df = pd.read_csv(schulzePath, header = None)
                Data_BH = df[0]
                Data_DC = df[1]
                DC = np.zeros_like(self.SMBH_mass)
                getDC = sp.interpolate.interp1d(Data_BH, Data_DC)
                fit = np.polyfit(Data_BH, Data_DC, 1)
                DC[self.SMBH_mass < np.amin(Data_BH)] = getDC(np.amin(Data_BH)) # )fit[0]*self.SMBH_mass[self.SMBH_mass < np.amin(Data_BH)] + fit[1]
                DC[self.SMBH_mass > np.amax(Data_BH)] = getDC(np.amax(Data_BH))
                slice = (self.SMBH_mass > np.amin(Data_BH)) * (self.SMBH_mass < np.amax(Data_BH))
                DC[slice] = getDC(self.SMBH_mass[slice])
                self.dutycycle = 10**DC
            else:
                assert False, "Unknown Duty Cycle Type {}".format(function)
        else:
            assert False, "No duty cycle type specified"

        assert self.dutycycle.any() >= 0.0, "DutyCycle elements < 0 exist. This is a probabiliy, and should therefore not valid"
        assert self.dutycycle.any() <= 1.0, "DutyCycle elements > 1 exist. This is a probabiliy, and should therefore not valid"

    def assignEddingtonRatios(self, type = "Schechter", redshift_evolution = False, knee = -1, alpha = -0.65):
        """Function to assign Eddington ratios and luminosities.

        Attributes:
            type (string) : The type of function to use to assign the luminosity.
                default is "Schechter", can be set to "PowerLaw"
            redshift_evolution (bool) : flag to add a redshift evolution term to
                function assigning luminosity
            knee (float) : value of the knee of this function. Default -1
            alpha (float) : value of alpha in this function, default -0.65
        """
        print("Assigning Eddington Ratios")

        def Schefunc(edd, z):
            prob = np.ones((len(edd)))

            gammaz = 3.47
            gammaE = alpha
            z0 = 0.6
            #knee = -1
            A = 10.**(-1.41)
            prob = ((edd/(10.**(knee)))**gammaE)

            if redshift_evolution:
                prob *= ((1.+z)/(1.+z0))**gammaz

            if type == "Schechter":
                return prob * np.exp(-(edd/(10.**(knee))));
            elif type == "PowerLaw":
                return prob
            else:
                assert False, "Type is unknown"

        self.TestForRequiredVariables(["SMBH_mass"])
        #Ledd = 1.28e38*(10.**self.SMBH_mass)
        Ledd = 38.1072 + self.SMBH_mass

        eddbin = np.arange(-4, 1, 0.0001, dtype = float)
        probSche = Schefunc(10**eddbin, self.z)

        p = probSche * 10**0.0001
        r_prob = p[::-1]
        probcum = np.cumsum(r_prob)
        r_probcum = probcum[::-1]
        y = r_probcum/r_probcum[0]
        y = y[::-1]
        eddbin = eddbin[::-1]

        a = np.random.random(len(self.SMBH_mass))
        lgedd = np.interp(a, y, eddbin) #, right=-99)
        Lbol = lgedd + Ledd


        #edd = 10.**lgedd
        #Lbol = (edd)*Ledd
        #lgLbol = np.log10(Lbol) - 33.49

        lgLbol = Lbol - 33.49
        lglum = lgLbol - 1.54 - (0.24*(lgLbol-12.)) - (0.012*((lgLbol-12.)**2.)) + (0.0015*((lgLbol-12.)**3.))
        lglum = lglum  + 33.49

        self.luminosity = lglum

        # Save Luminosity Function Data
        step = 0.1
        bins = np.arange(42, 46, step)
        Lum_bins = sp.stats.binned_statistic(self.luminosity, self.dutycycle, 'sum', bins = bins)[0]
        Lum_func = (Lum_bins/self.volume)/step
        self.XLF_plottingData.append(PlottingData(bins[0:-1][Lum_func > 0], np.log10(Lum_func[Lum_func > 0])))

        # Save Eddington Distribution Data
        step = 0.5
        Lg_edd_derived = np.log10(25) + self.luminosity - (35.3802 + self.stellar_mass)  #log10(1.26e38 * 0.002) = 35.3802
        eddbin = np.arange(-4, 1, step)
        prob_derived = stats.binned_statistic(Lg_edd_derived, self.dutycycle, 'sum', bins = eddbin)[0]/(step * sum(self.dutycycle))

        eddbin = eddbin[:-1]
        eddbin = eddbin[prob_derived > 0]
        prob_derived = prob_derived[prob_derived > 0]

        self.Edd_plottingData.append(PlottingData(eddbin, np.log10(prob_derived)))

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
        self.TestForRequiredVariables(["dutycycle", "luminosity", "x_coord", "y_coord", "z_coord"])
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

    def Obscuration(self, ret = "Obscured", Obscured = 23):
        print("Calculating Obscuration")
        """Function to assign column density, Nh,

        Also cuts the catalogue depending on the specified value of Nh

        No attributes
        """
        self.TestForRequiredVariables(["lum_cat"])

        lgNH = np.arange(20., 30., 0.0001)  # Range, or effective 'possible values' of Nh

        lgNHAGNSch1 = np.ones(len(self.lum_cat))

        lgLxbin = np.arange(np.amin(self.lum_cat), np.amax(self.lum_cat), 0.01)

        for i in range(len(lgLxbin)-1): # index for Lx bins

            flag  = np.where((self.lum_cat >=lgLxbin[i]) & (self.lum_cat < lgLxbin[i+1])) # Boolean flag for when  arrays are lower than iterated value
            #flag = lightningWhere(self.luminosity, lgLxbin[i], lgLxbin[i+1])

            fLxzNH  = (NHfunc(lgLxbin[i], self.z, lgNH))  # call fn
            fLxzNH = fLxzNH * 0.01
            fLxzNH_= fLxzNH[::-1] # reverse
            fLxzNH_cum = np.cumsum(fLxzNH_) # cumulative sum
            r_fLxzNH_cum  = fLxzNH_cum[::-1] # Reverse again
            y = r_fLxzNH_cum/ r_fLxzNH_cum[0] # Normalize(ish?) by the first element
            y = y[::-1]  # Reverse AGAIN
            lgNH = lgNH[::-1] # R
            a = np.random.random()

            lgNHAGNSch1[flag] = np.interp(a, y, lgNH, right=-99)

        N_h = lgNHAGNSch1

        self.obscured = N_h > Obscured

        if ret == "Obscured":
            N_h_Obscured_Flag = N_h > Obscured
            print("    {}% of the remaining catalogue remains as obscured".format(np.round(100 * np.sum(N_h_Obscured_Flag)/len(N_h_Obscured_Flag), 2)))
            self.x_cat = self.x_cat[N_h_Obscured_Flag]
            self.y_cat = self.y_cat[N_h_Obscured_Flag]
            self.z_cat = self.z_cat[N_h_Obscured_Flag]
            self.dc_cat = self.dc_cat[N_h_Obscured_Flag]
        elif ret == "Unobscured":
            N_h_unobscured_Flag = N_h < Obscured
            print("    {}% of the remaining catalogue remains as unobscured".format(np.round(100 * np.sum(N_h_unbscured_Flag)/len(N_h_unbscured_Flag), 2)))
            self.x_cat = self.x_cat[N_h_unobscured_Flag]
            self.y_cat = self.y_cat[N_h_unobscured_Flag]
            self.z_cat = self.z_cat[N_h_unobscured_Flag]
            self.dc_cat = self.dc_cat[N_h_unobscured_Flag]
        else:
            assert False, "Unknown argument name {}, should be Obscured or Unobscured".format(ret)

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
        self.TestForRequiredVariables(["x_cat", "y_cat", "z_cat", "lum_cat"])
        if threads == "System":
            threads = multiprocessing.cpu_count()
        period = self.volume**(1/3)
        rbins = np.logspace(-1, 1.5, 50)
        self.wpbins = rbins
        wp_results = wp(period, pi_max, threads, rbins,\
                self.x_cat, self.y_cat, self.z_cat, weights = self.dc_cat, weight_type = 'pair_product', verbose = True)
        xi = wp_results['wp']

        self.WP_plottingData.append(PlottingData(rbins[:-1], xi))

    def computeBias(self, variable, binsize = 0.3, weight = True, mask = None):
        # Functions borrowed from Viola's Code - I'm not entirely sure what they all do.
        requiredVariables = ["parent_halo_mass"]
        if weight:
            requiredVariables.append("dutycycle")
        self.TestForRequiredVariables(requiredVariables)

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
        bias = estimate_biasTin(((10.**self.parent_halo_mass)*0.974)*self.h, 0.25, g_squared) # So this is the bias of the haloes?

        bin = np.arange(np.amin(variable), np.amax(variable), binsize)
        meanbias = np.zeros_like(bin)
        errorbias = np.zeros_like(bin)

        for i in range(len(bin)-1):
            N1 = np.where(((variable) >= bin[i]) & ((variable) < bin[i+1]))
            if mask != None:
                N1 *= mask # Mask out for obscured etc if we want to.
            if weight:
                meanbias[i] = np.sum(bias[N1]*self.dutycycle[N1])/np.sum(self.dutycycle[N1])
                errorbias = np.sqrt((np.sum(self.dutycycle[N1] * (bias[N1] - meanbias[i])**2))/(((self.dutycycle[N1] - 1)/len(self.dutycycle[N1])) * np.sum(self.dutycycle[N1])))
            else:
                meanbias[i] = np.mean(bias[N1])
                errorbias = np.std(bias[N1])

        self.bias_plottingData.append(PlottingData(bin, meanbias, errorbias))



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

class PlottingData:
    def __init__(self, x, y, error = None):
        self.x = x
        self.y = y

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
        h = self.h
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



def GetCorrectFile(string, redshift, directory = "./", retz = False):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    directories = os.listdir(directory)

    list = [];
    string_list = [];
    for file in directories: # Go through all strings (files) in the directory.
        if string in file: # If the string is in the filename
            stripped = file.replace(string, '') # remove the string in case we need to
            res = rx.findall(stripped) # Find the numbers within the string.
            if res != []:
                list.append(float(res[0]))
                string_list.append(file)

    assert len(list) >= 1, "Files containing {} were not found".format(string)

    index = (np.abs(np.array(list) - redshift)).argmin()

    diff = abs(redshift - list[index])
    if diff > 0.1:
        print("Warning - we have requested redshift {} - Selecting file {} as it is closest".format(redshift, string_list[index]))
    if retz:
        return string_list[index], list[index]
    else:
        return string_list[index]

def ReadSimpleFile(string, redshift, path):
    file = path + GetCorrectFile(string, redshift, path)
    df = pd.read_csv(file, header = None)
    return df[0], df[1]



if __name__ == "__main__":
    default = AssignmentVariation()
    default.setRedshift(1.5)
    default.generateSemiAnalyticHaloes(volume = 250**3)
    default.assignStellarMass()
    default.assignBlackHoleMass()
    default.assignEddingtonRatios()
    default.assignDutyCycle(function = "Constant")
    a, b = default.getLuminosityFunction()
    c, d = default.getEddingtonDistribution()
