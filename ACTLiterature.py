import numpy as np
# Local
from ACTUtillity import *


def aird_edd_dist(edd, z):
    """ Function to return the Eddington Ratio distribution from Aird.

    :param edd: array, the eddington ratio bins.
    :param z: float redshift
    :return: array, the probability.
    """
    gamma_e = -0.65
    gamma_z = 3.47
    z0 = 0.6
    a = 10.**(-3.15)
    prob = a*((edd/(10.**(-1.)))**gamma_e)*((1.+z)/(1.+z0))**gamma_z
    return prob


class WPData(data):
    """ Class to store comparison data for wp.

    :param z: float, the redshift.
    """
    def __init__(self, z):
        data.__init__(self, z)

        self.r_Koutoulidis = np.array([0.2, 0.35, 0.55, 1, 1.5, 2.5, 3.4, 7.0, 20])
        self.wp_Koutoulidis = np.array([11, 175, 47, 45, 37, 25, 25, 11, 6.1])
        self.wp_Koutoulidis_e = np.array([[50, 100, 20, 10, 10, 1, 1, 1, 1],
                                          [50, 100, 20, 10, 10, 1, 1, 1, 1]])

    def k_power_law(self, rp, r0=6.2, gam=1.88):
        rp = rp/self.h
        wp = rp * ((r0/rp) ** gam) * ( (special.gamma(0.5) * special.gamma((gam-1)/2)) / special.gamma(gam/2))
        return wp * self.h


class XLFData(data):
    """ Class to store the comparison data for XLF

    :param z: float, the redshift
    """
    def __init__(self, z):
        data.__init__(self, z)
        # Read in Miyanji
        mi_lx, mi_phi = ReadSimpleFile("Miyaji2015", self.z, self.dataPath)
        self.mi = PlottingData(10**mi_lx, mi_phi)

    def get_miyaji2015(self):
        """ Returns the plotting data for miyanji2015

        :return: PlottingData Object, with Lx vs phi.
        """
        return self.mi

    def get_ueda14(self, bins, nh='free'):
        """ Return LF in the 2-10 KeV range based on the article from Ueda et al. (2014) in this version of the program
        we also take into account the detailed distribution of the Nh column density with L and z and also the possible
        contribution by Compton-thick AGNs.

        :param: bins, array, the bins of luminosity.
        :return: PlottingData objects with bins vs phi.
        """
        luminosity_bins = bins
        z = self.z

        # Constants
        a = 2.91e-6
        l_s = 43.97
        l_p = 44.
        g_1 = 0.96
        g_2 = 2.71
        p_1s = 4.78
        p_2 = -1.5
        p_3 = -6.2
        b_1 = 0.84
        z_sc1 = 1.86
        l_a1 = 44.61
        alpha_1 = 0.29
        z_sc2 = 3.0
        l_a2 = 44.
        alpha_2 = -0.1

        # Preallocate Memory
        nl = len(luminosity_bins)
        z_c1 = np.zeros(nl)
        z_c2 = np.zeros(nl)
        e = np.zeros(nl)
        l_x = luminosity_bins.copy()

        z_c1[l_x <= l_a1] = z_sc1 * (10.**(l_x[l_x < l_a1] - l_a1))**alpha_1
        z_c1[l_x > l_a1] = z_sc1
        z_c2[l_x <= l_a2] = z_sc2 * (10.**(l_x[l_x <= l_a2]-l_a2))**alpha_2
        z_c2[l_x > l_a2] = z_sc2

        p_1 = p_1s + b_1*(l_x - l_p)

        e[z <= z_c1] = (1 + z)**p_1[z <= z_c1]
        e[(z > z_c1) & (z < z_c2)] = (1. + z_c1[(z > z_c1) & (z < z_c2)])**p_1[(z > z_c1) & (z < z_c2)] * ((1.+z)/(1.+z_c1[(z > z_c1) & (z < z_c2)]))**p_2
        e[z > z_c2] = (1. + z_c1[z > z_c2])**p_1[z > z_c2] * ((1.+z_c2[z > z_c2])/(1.+z_c1[z > z_c2]))**p_2 * ((1. + z)/(1. + z_c2[z > z_c2]))**p_3

        den1 = (10.**(l_x - l_s))**g_1
        den2 = (10.**(l_x - l_s))**g_2
        den = den1 + den2

        phi = (a/den) * e

        bet = 0.24
        a1 = 0.48
        f_ctk = 1.0
        psi0 = 0.43

        if z < 2:
            psi44z = psi0 * (1.0 + z)**a1
        else:
            psi44z = psi0 * (1.+2.)**a1

        eta = 1.7
        psi_max = (1. + eta)/(3. + eta)
        psi_min = 0.2

        em = psi44z - bet*(luminosity_bins - 43.75)

        psi = np.ones(nl) * psi_min
        psi[em > psi_min] = em[em > psi_min]
        psi[psi_max > psi] = psi_max

        if nh == 'free':
            return PlottingData(10**bins, phi)

        lim = (1. + eta)/(3. + eta)
        frac = np.zeros(nl)

        for k in range(nl):
            if psi[k] < lim:
                if nh == 0:
                    frac[k] = 1. - (2. + eta)/(1. + eta)*psi[k]   # 20.<LogNh<21
                if nh == 1:
                    frac[k] = (1./(1. + eta))*psi[k]          # 21.<LogNh<22.
                if nh == 2:
                    frac[k] = (1./(1. + eta))*psi[k] 			# 22.<LogNh<23.
                if nh == 3:
                    frac[k] = (eta/(1.+eta))*psi[k] 	    # 23.<LogNh<24.
                if nh == 4:
                    frac[k] = (f_ctk/2.)*psi[k] 				# 24.<LogNh<26
            else:
                if nh == 0:
                    frac[k] = 2./3. - (3. + 2.*eta)/(3. + 3.*eta)*psi[k]  # 20.<LogNh<21
                if nh == 1:
                    frac[k] = 1./3. - eta/(3. + 3.*eta)*psi[k]          # 21.<LogNh<22.
                if nh == 2:
                    frac[k] = (1./(1. + eta))*psi[k] 					# 22.<LogNh<23.
                if nh == 3:
                    frac[k] = (eta/(1. + eta))*psi[k] 			    # 23.<LogNh<24.
                if nh == 4:
                    frac[k] = (f_ctk/2.)*psi[k] 						# 24.<LogNh<26
        phi = frac*phi
        return PlottingData(10**bins, phi)


class EddingtonDistributionData(data):
    """ Class to store the comparison data for the Eddington Ratio distribution.

    :param z: float, redshift.
    """
    def __init__(self, z, datapath="./Data/"):
        data.__init__(self, z, datapath=datapath)

        # Read in Geo17
        geo_lx, geo_phi_top, geo_phi_bottom, z = ReadSimpleFile("Geo17", self.z, self.dataPath, cols=3, retz=True)
        self.Geo = IntervalPlottingData(geo_lx, geo_phi_top, geo_phi_bottom, z=z)

        # Read in Bon 16
        bon16_lx, bon16_phi, z = ReadSimpleFile("Bon16", self.z, self.dataPath, retz=True)
        self.Bon16 = PlottingData(bon16_lx, bon16_phi, z=z)

        # Read in Bon 12
        bon12_lx, bon12_phi, z = ReadSimpleFile("Bon12", self.z, self.dataPath, retz=True)
        self.Bon12 = PlottingData(bon12_lx, bon12_phi, z=z)

        # Aird
        aird_lx, aird_phi, z = ReadSimpleFile("Aird2018", self.z, self.dataPath, retz=True)
        self.Aird = PlottingData(aird_lx, aird_phi, z=z)

        # Aird12
        aird_lx, aird_phi, z = ReadSimpleFile("Aird12", self.z, self.dataPath, retz=True)
        self.Aird12 = PlottingData(aird_lx, aird_phi, z=z)

    def AirdDist(self, edd):
        gamma_e = -0.65
        gamma_z = 3.47
        z0 = 0.6
        A = 10.**(-3.15)
        prob = A*((10**edd/(10.**(-1.)))**gamma_e)*((1.+self.z)/(1.+z0))**gamma_z

        bin_width = edd[1] - edd[0]
        prob /= bin_width # divide by bin width
        prob = np.log10(prob[prob > 0])
        return prob


if __name__ == "__main__":
    EddData = EddingtonDistributionData(0.0)