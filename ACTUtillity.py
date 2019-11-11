# General
import numpy as np
import scipy as sp
from scipy import stats
from scipy import special
import pandas as pd
import os
import glob
import re
import sys
import multiprocessing
from numba import jit
from math import pi
# Specialized
from colossus.cosmology import cosmology

def GetCorrectFile(string, redshift, directory_path ="./", retz = False):
    """ Function to find the best file in a directory given a redshift.

    Looking inside the directory, this function looks for file containing 
    the supplied string, and then finds a numerical component that represents 
    the redshift. For example, supplying "MD_" and "0.5" would first find all 
    files with "MD_" in their name, then the one with z closest to 0.5.

    Arguments:
        string (string) : The (non-numeric) component of the filename.
        redshift (float) : The desired redshift.
        directory_path (string) : Path to the directory we want to look in. Default
            is the current directory.
        retz (bool) : if set to True, returns the found redshift as a second 
            argument. Defaults to False.
    Returns:
        The filename as a string, and if retz = True, the redshift of the file
        as a float.
    """
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    directories = os.listdir(directory_path)

    list = []
    string_list = []
    for file in directories: # Go through all strings (files) in the directory.
        if string in file and file[0] != '.': # If the string is in the filename
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

def ValidatePath(path, ErrorOnFail = False):
    """Simple function to check if a path exists, and if not create it.

    Arguments:
        path (string) : the desired path
    Returns:
        path (string) : just for utility
    """
    if not os.path.isdir(path):
        if ErrorOnFail:
            assert False, "{} does not exist, and most likely contains files needed for a successful run".format(path)
        os.mkdir(path)
    return path 

def ReadSimpleFile(string, redshift, path, cols=2, retz=False):
    """Function to find and read a two column CSV (normally data).

    Operation is similar to GetCorrectFile, but actually reads the CSV and 
    returns in as two columns.

    Arguments:
        string (string) : The (non-numeric) component of the filename.
        redshift (float) : The desired redshift.
        path (string) : Path to the directory we want to look in. Default
            is the current directory.
        cols (int) : default 2, the number of columns in the file.
    Returns:
        The columns of the CSV, in two parameters (A and B) as slices of
        pandas dataframes. If cols = 3, will return 3 parameters.
    """
    file_correct, z = GetCorrectFile(string, redshift, path, retz=True)
    file = path + file_correct
    df = pd.read_csv(file, header=None)
    print(df)
    if cols == 2:
        if retz:
            return df[0].values, df[1].values, z
        return df[0].values, df[1].values
    elif cols == 3:
        if retz:
            return df[0].values, df[1].values, df[2].values, z
        return df[0].values, df[1].values, df[2].values

def TestForRequiredVariables(Obj, Names):
    """Function to check that the supplied list of variables actually exist.

    This is for internal use. Will throw an assertion error if names do not
    exist.

    Arguments:
        Names (array of strings) : the names of the variables to search for.
    """
    for name in Names:
        assert hasattr(Obj, name), "You need to assign {} first".format(name)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def erase_all_in_folder(path):
    assert path[0] == '.', "For safety relative paths only"
    assert path[-1] == '/', "Directories only"
    path += '*'
    files = glob.glob(path)

    if len(files) != 0:
        print("Some files will be automatically removed, listed below.")
        for f in files:
            print("{}".format(f))

        assert query_yes_no("Are you okay with this?"), "Aborting"

        print("Removing Files")
        for f in files:
            os.remove(f)


def visual_debugging_housekeeping(visual_debugging=True,
                                  function_name="default",
                                  erase_debugging_folder=False,
                                  visual_debugging_path="./default/"):
    """ function to encapsulate the visual debugging housekeeping that a lot of the functions have to do.

    :param visual_debugging: bool, if to actually continue with visual debugging. If False, this function stops dead.
    :param function_name: string, the name of the function purely used for printing to the terminal
    :param erase_debugging_folder: bool, if True will erase the entire contents of folder supplied below
    :param visual_debugging_path: string, the path to the folder.
    :return:
    """
    if visual_debugging:
        print("You have activated visual debugging for {}".format(function_name))
        ValidatePath(visual_debugging_path)
        if erase_debugging_folder:
            erase_all_in_folder(visual_debugging_path)


class PlottingData:
    def __init__(self, x, y, error=None, z=0):
        self.x = x
        self.y = y
        self.z = z


class IntervalPlottingData:
    def __init__(self, x, yu, yd, z=0):
        self.x = x
        self.yu = yu
        self.yd = yd
        self.z = z


class data:
    # Parent class for data type objects
    def __init__(self, z):
        self.dataPath = "./Data/"
        self.z = z
        self.cosmology = cosmology.setCosmology('planck18')
        self.h = self.cosmology.H0/100

if __name__ == "__main__":
    # For the testing of these specific functions.

    directory = "./Testing/Utility/"

    def RedshiftTest(find, compare):
        A, z = GetCorrectFile("Tes", find, directory, retz = True)
        RedshiftFail = "GetCorrectFile is not returning correct z: should be {}, returning {}"
        assert z == compare, RedshiftFail.format(find, z)

    # Files that do exist, and a 1-1 comparison should be fine
    testz = [0., 0.5, 0.75, 1.0, 1.25]
    for test in testz:
        RedshiftTest(test, test)
    # Files that don't exist, so we make sure they get the closest
    testz2 = [0.2, 0.6, 0.74999, 5, -1]
    should = [0.0, 0.5, 0.75, 1.25, 0.0]
    for i, test in enumerate(testz2):
        RedshiftTest(test, should[i])
    
    A, B = ReadSimpleFile("Dat", 0.0, directory)

    correct = np.array([[0., 1.], [2, 3]])

    A2 = correct[:, 0]
    B2 = correct[:, 1]
    ReadFail = "ReadSimpleFile Failed - should be returning {}, but returned {}"
    assert (A2 == np.array(A)).all(), ReadFail.format(A2, np.array(A))
    assert (B2 == np.array(B)).all(), ReadFail.format(B2, np.array(B))

    print("Tests passed Sucessfully")
