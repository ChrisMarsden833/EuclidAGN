import ACTTestingEncapsulation as actt
import ACTLiterature as actl 

import numpy as np
import os
import os.path
from os import path

from scipy import interpolate
import datetime

class Fitter:
    def __init__(self, redshift):
        self.redshift = redshift
        # Create the master class.
        self.obj = actt.AGNCatalog(print_updates=False)
        self.obj.set_z(redshift)
        # Load in the MultiDark Haloes.
        self.obj.load_dm_catalog(path_big_data="/data/cm1n17/MultiDark/npyData/")
        self.obj.assign_stellar_mass()
        self.obj.assign_black_hole_mass()
        self.obj.assign_duty_cycle("Schulze")
        # Cut Volume for convnience
        limit = 700  # Mpc/h
        flag = (self.obj.main_catalog['x'] < limit) * (self.obj.main_catalog['y'] < limit) * (self.obj.main_catalog['z'] < limit)

        self.obj.main_catalog = self.obj.main_catalog[flag]
        self.obj.volume = (limit / self.obj.h) ** 3

        self.EddData = actl.EddingtonDistributionData(self.redshift)

        # XLF Stuff
        self.XLF = actl.XLFData(self.redshift)
        # XLF Data
        self.uXLF_data = self.XLF.get_ueda14(np.arange(44, 45, 0.1))


    def DoAGNIteration(self, alphaValue = 0.5, lambda_array = np.linspace(-1, 1, 2)):
        variations = lambda_array
        iterations = len(variations)

        for i in range(iterations):
            self.obj.assign_luminosity(parameter1=variations[i], parameter2=alphaValue) # Lambda Shape? # alpha Normalization

        XLF_RMS = np.zeros_like(lambda_array)

        for i in range(iterations):
            #RMS comparison to Udea
            xto_data = interpolate.interp1d(self.obj.XLF_plottingData[i].x, self.obj.XLF_plottingData[i].y, bounds_error=False, fill_value=((self.obj.XLF_plottingData[i].y)[0], (self.obj.XLF_plottingData[i].y)[-1]))
            xto_comparison = interpolate.interp1d(np.log10(self.uXLF_data.x), np.log10(self.uXLF_data.y), bounds_error=False, fill_value=( ( np.log10(self.uXLF_data.y[0]), np.log10(self.uXLF_data.y[-1]) ) ) )

            x_range = np.linspace(42, 45)
            XLF_RMS[i] = np.sqrt( np.sum( ( (xto_data(x_range) - xto_comparison(x_range) )**2 ) ) )

        # ERD Stuff

        ERD_RMS = np.zeros_like(lambda_array)

        for i in range(iterations):
            # RMS
            xto_data = interpolate.interp1d(self.obj.Edd_plottingData[i].x, self.obj.Edd_plottingData[i].y, bounds_error=False, fill_value=((self.obj.Edd_plottingData[i].y)[0], (self.obj.Edd_plottingData[i].y)[-1]))

            xto_top = interpolate.interp1d(self.EddData.Geo.x, self.EddData.Geo.yu)
            xto_bottom = interpolate.interp1d(self.EddData.Geo.x, self.EddData.Geo.yd)

            x_range = np.linspace(-3.5, -0.5)

            sample_data = xto_data(x_range)
            sample_u = xto_top(x_range)
            sample_d = xto_bottom(x_range)

            res = np.zeros_like(x_range)

            res[sample_data > sample_u] = sample_data[sample_data > sample_u] - sample_u[sample_data > sample_u]
            res[sample_data < sample_d] = sample_data[sample_data < sample_d] - sample_d[sample_data < sample_d]
            ERD_RMS[i] = np.sqrt(np.sum(res**2))

        self.obj.Edd_plottingData = []
        self.obj.XLF_plottingData = []

        return XLF_RMS, ERD_RMS


if __name__ == "__main__":

    if path.exists('FittingResults.txt'):
        os.remove('FittingResults.txt')

    # Create file
    with open('FittingResults.txt', 'a') as the_file:
        the_file.write('Commenced: {}\n'.format(datetime.datetime.now()))

    z_array = [2]

    for z in z_array:
        alpha_range = np.arange(0., 20., 1.)
        lambda_range = np.arange(-8., 5., 1.)

        print("Lambda range", lambda_range)

        XLFRMS_minima = np.zeros_like(alpha_range)
        ERDRMS_minima = np.zeros_like(alpha_range)
        lamda_values = np.zeros_like(alpha_range)
        corresponding_lamda_index = np.zeros_like(alpha_range)

        obj = Fitter(z)

        for i, element in enumerate(alpha_range):
            print("z = {}, {} percent".format(z, 100*i/len(alpha_range) ))
            XLF_RMS, ERD_RMS = obj.DoAGNIteration(alphaValue=element, lambda_array=lambda_range)

            min_index = np.argmin(XLF_RMS)

            print("Alpha: {}, Lambda_range: {}, XLF_RMS {}, min: {}".format(element, lambda_range, XLF_RMS, min_index))

            XLFRMS_minima[i] = XLF_RMS[min_index]
            ERDRMS_minima[i] = ERD_RMS[min_index]
            lamda_values[i] = lambda_range[min_index]
            corresponding_lamda_index[i] = min_index

        super_min_index = np.argmin(XLFRMS_minima)

        print("z = {}, alpha = {}, lambda = {}".format(z, alpha_range[super_min_index],
                                                       lambda_range[int(corresponding_lamda_index[super_min_index])] ) )

        with open('FittingResults.txt', 'a') as the_file:
            the_file.write("z = {}, alpha = {}, lambda = {}, time = {}\n".format(z, alpha_range[super_min_index],
                                                                                 lambda_range[int(corresponding_lamda_index[super_min_index])], datetime.datetime.now()))

        print(datetime.datetime.now())

