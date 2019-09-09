import numpy as np
import scipy as sp
from scipy import stats
from scipy import special
import pandas as pd
import os
import re
import multiprocessing
from numba import jit
from math import pi


def GetCorrectFile(string, redshift, directory = "./", retz = False):
    """ Function to find the best file in a directory given a redshift.

    Looking inside the directory, this function looks for file containing 
    the supplied string, and then finds a numerical component that represents 
    the redshift. For example, supplying "MD_" and "0.5" would first find all 
    files with "MD_" in their name, then the one with z closest to 0.5.

    Arguments:
        string (string) : The (non-numeric) component of the filename.
        redshift (float) : The desired redshift.
        directory (string) : Path to the directory we want to look in. Default
            is the current directory.
        retz (bool) : if set to True, returns the found redshift as a second 
            argument. Defaults to False.
    Returns:
        The filename as a string, and if retz = True, the redshift of the file
        as a float.
    """
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    directories = os.listdir(directory)

    list = []
    string_list = []
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
    """Function to find and read a two column CSV (normally data).

    Operation is similar to GetCorrectFile, but actually reads the CSV and 
    returns in as two columns.

    Arguments:
        string (string) : The (non-numeric) component of the filename.
        redshift (float) : The desired redshift.
        path (string) : Path to the directory we want to look in. Default
            is the current directory.
    Returns:
        The columns of the CSV, in two parameters (A and B) as slices of
        pandas dataframes.
    """
    file = path + GetCorrectFile(string, redshift, path)
    df = pd.read_csv(file, header = None)
    return df[0], df[1]

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