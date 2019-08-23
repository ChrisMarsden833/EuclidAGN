# Generic
import numpy as np
import warnings
import difflib
from matplotlib import pyplot as plt

# Local
from DynamicTesting import *


class VariationManager:

    def __init__(self, parameters):

        Obj = EuclidMaster()

        function_lookup = {"Redshift" : Obj.setRedshift,
                        "Dark Matter" : Obj.DarkMatter,
                       "Stellar Mass" : Obj.assignStellarMass,
                    "Black Hole Mass" : Obj.assignBlackHoleMass,
                         "Duty Cycle" : Obj.assignDutyCycle,
                   "Eddington Ratios" : Obj.assignEddingtonRatios,
                          "Catalogue" : Obj.CreateCatalogue,
                        "Obscuration" : Obj.Obscuration,
                         "Clustering" : Obj.computeWP }

        argument_numbers = [1,
                            1,
                            1,
                            3,
                            1,
                            4,
                            3,
                            2,
                            1]

        redshift = parameters["Redshift"]

        function_list = []
        argument_no_list = []
        name_param_lookup = {}

        for i, parameter in enumerate(parameters): # Loop over all the specified parameter Names
            if parameter in function_lookup: # When the name appears in the function lookup
                function_list.append(function_lookup[parameter]) # Append the function to the list of functions to be later executed
                argument_number = argument_numbers[i] # Find the approprate number of arguments for this function from the lookup
                argument_no_list.append(argument_number) # Append this to the list of approprate arguments corresponding to the functions to be called.

                #if (argument_number > 1) and (type(parameters[parameter][0]) is not list): # These lines just allow the user the option of having two write a single list as a list within a list
                parameters[parameter] = fixList(parameters[parameter])
                name_param_lookup[parameter] = parameters[parameter].copy()

                for i, element in enumerate(parameters[parameter]): # Loop over all the iterations
                    supplied = len(fixList(element)) # Get the length of the argument(s) - the number of suppled arguments for one iteration
                    if len(fixList(parameters[parameter])) > 1:
                        positionInfo = " at position {}".format(i) # if conditions just to get error formatting right
                    else:
                        positionInfo = ""
                    assert supplied == argument_number, "Number of arguments supplied for {} ({} supplied{}) does not match the required number ({}).".format(parameter, i, positionInfo, argument_number)
            else: # Unrecognised parameter logic
                close = difflib.get_close_matches(parameter, function_lookup)
                if len(close) > 0:
                    assert False, "Method name {} was not recognized, but it is close to: {}".format(parameter, close)
                else:
                    assert False, "Method name {} was not recognized"

        iterations, counts = DictionaryProduct(parameters)
        print("iterations: ", iterations)
        start_repeats = first_nonunity(counts)

        combination_matrix = constructCombinationsMatrix(counts)
        print(combination_matrix)


        start = 0
        first = True
        for i in range(iterations):
            map = combination_matrix[:, i]
            for j, data in enumerate(parameters.values()):
                if not first and j < start_repeats:
                    pass
                else:
                    version = map[j] - 1
                    sub = data[int(version)]
                    sub = fixList(sub)
                    function_list[j](*sub)
            first = False


        # Prepare titles
        title_array = []
        legend_array = []
        for i in range(iterations):
            map = combination_matrix[:, i]
            title = ""
            legend = ""
            for j, name in enumerate(name_param_lookup):
                if counts[j] == 1:
                    title += " {}: {},".format(name, name_param_lookup[name][map[j]-1])
                else:
                    legend += " {}: {},".format(name, name_param_lookup[name][map[j]-1])
            title_array.append(title)
            legend_array.append(legend)

        # TODO split this into a second function call so we can make changes to it if necessary

        # Plot of the XLF
        plt.figure()

        for i in range(iterations):
            plt.plot(10**Obj.XLF_plottingData[i].x, 10**Obj.XLF_plottingData[i].y, label = legend_array[i])

        # XLF Data
        XLF = XLF_Data(redshift)
        plt.plot(10**XLF.Mi_LX, 10**XLF.Mi_phi, 'o', label = "Miyaji")

        uXLF, ubins = XLF.getUeda14(np.arange(42, 46, 0.1))
        plt.plot(10**ubins, 10**uXLF, ':', label = "Ueda")

        # Plotting
        plt.xlabel(r'$L_x\;[erg\;s^{-1}]$')
        plt.ylabel(r'$d\phi /d(log\;L_x)\;[Mpc^{-3}]$')
        plt.title(title_array[0])
        #r'XLF, z = {}, \lambda = {}, alpha = {}'.format(redshift, -1, -0.65), fontname = 'Times New Roman')
        plt.loglog()
        plt.legend()
        plt.show()

        # Plot the Eddington ratio distribution

        plt.figure()

        for i in range(iterations):
            plt.plot(Obj.Edd_plottingData[i].x, Obj.Edd_plottingData[i].y, ':', label = legend_array[i])

        EddData = EddingtonDistributionData(redshift)
        plt.plot(EddData.Geo_LX, EddData.Geo_phi, label = "Geo")
        plt.plot(EddData.Bon16_LX, EddData.Bon16_phi, label = "Bon 16")
        plt.plot(EddData.Bon12_LX, EddData.Bon12_phi, label = "Bon 12")

        eddbin = np.arange(-4, 1., 0.5)
        probSche = EddData.AirdDist(eddbin)
        plt.plot(eddbin, probSche, label = "Aird (analytic)")
        plt.title(title_array[0])
        #r'Eddington Ratio Distribution, z = {}, U = {}'.format(redshift, dutycycle), fontname = 'Times New Roman')
        plt.xlabel("Eddington Ratio")
        plt.ylabel("Probability $log10$")

        plt.legend()
        plt.show()

        plt.figure()

        wp_data = WP_Data(redshift)

        for i in range(iterations):
            plt.plot(Obj.WP_plottingData[i].x, Obj.WP_plottingData[i].y, ':', label = legend_array[i])

        plt.errorbar(wp_data.r_Koutoulidis, wp_data.wp_Koutoulidis,\
                        yerr = wp_data.wp_Koutoulidis_e, fmt='o', label = "Koutoulidis Data")

        plt.plot(obj.wpbins, wp_data.K_powerLaw(Obj.wpbins), label = "Koutoulidis Fit")
        plt.title(title_array[0])
        #r'wp, z = {}, U = {}'.format(redshift, dutycycle), fontname = 'Times New Roman')
        plt.xlabel(r'$r_p$ $Mpc$')
        plt.ylabel(r'$w(r_p)$')

        plt.legend()
        plt.loglog()
        plt.show()










def DictionaryProduct(dictionary):
    final = 1
    counts = np.ones(len(dictionary))
    for i, value in enumerate(dictionary.values()):
        value = fixList(value)
        counts[i] = len(value)
        final *= len(value)
    return final, counts

def fixList(entity):
    if type(entity) is not list:
        entity = [entity]
    return entity

def first_nonunity(array):
    for i, element in enumerate(array):
        if element > 1:
            return i
    return None

def constructCombinationsMatrix(array):
    dim = len(array)
    combinations = int(np.prod(array))
    holder = ()
    for i in range(dim):
        element = array[i]
        temp = []
        for j in range(int(element)):
            temp.append(j + 1)
        holder += (temp,)

    holder1 = cartesian(holder)
    return(holder1.T)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out



"""

def fixList(entity):
    if type(entity) is not list:
        entity = [entity]
    return entity

def DictionaryProduct(dictionary):
    final = 1
    for value in dictionary.values():
        final *= len(value)
    return final

class EuclidObject:
    def __init__(self, cosmology = 'planck18'):
        # Prep
        cosmology = fixList(cosmology)

        self.masterObjectList = [] # Object list

        self.VariationNames = cosmology.copy() # List of lists, showing the names of all the variations that were called.
        for i, value in enumerate(cosmology):
            self.VariationNames[i] = [cosmology[i]] # Fix to get the list of lists working properly

        self.OrderingLookup = [] # Essentially the order in which the functions were called.
        self.NumberingLookup = {} # Dictionary corresponding to



        EntryName = "Cosmology"
        self.OrderingLookup.append(EntryName)
        self.NumberingLookup[EntryName] = len(cosmology)

        for i, entry in enumerate(cosmology):
            #self.VariationNames[i].append(entry)
            self.masterObjectList.append(EuclidMaster(cosmo = entry))

    def SetRedshift(self, redshift):

        # A lot of the verboseness of this section is to avoid python 'pointer hell'
        redshift = fixList(redshift)
        EntryName = "Redshift"
        length = len(redshift)
        self.OrderingLookup.append(EntryName)
        self.NumberingLookup[EntryName] = length

        assert len(self.VariationNames) == len(self.masterObjectList), "Length of master object list (length {}) != to the Variation Names (length {})".format(len(self.VariationNames), len(self.masterObjectList))
        original_length = len(self.VariationNames)

        temp_names_moving = []
        temp_objects_moving = []
        old_names_length = len(self.VariationNames)
        for i in range(length):
            for j in range(old_names_length):
                temp_names_moving.append(self.VariationNames[j].copy())
                temp_objects_moving.append(self.masterObjectList[j].copy())
        self.VariationNames = temp_names_moving.copy()



        self.masterObjectList = np.tile(self.masterObjectList, length)

        j = 0
        for i in range(len(self.VariationNames)):
            self.VariationNames[i].append(redshift[j])
            j += 1
            if j == length - 1:
                j = 0

    """
