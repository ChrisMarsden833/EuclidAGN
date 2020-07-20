from AGNCatalogToolbox import main as agn
from colossus.cosmology import cosmology
cosmology = cosmology.setCosmology('planck18')

volume = 200**3  # Relatively small cosmological volume for this example
z = 0

# Use the semi-analytic halo generator to create our preliminary catalog
halos = agn.generate_semi_analytic_halo_catalogue(volume, (12, 16, 0.1), z, cosmology.h)
# Stellar mass is the assigned to each of these values
stellar_mass = agn.halo_mass_to_stellar_mass(halos, z)
# Black hole mass & Duty Cycle are assigned using stellar mass
black_hole_mass = agn.stellar_mass_to_black_hole_mass(stellar_mass, method="Shankar16", scatter="Intrinsic")
duty_cycle = agn.to_duty_cycle(0.1, stellar_mass, black_hole_mass, z)

# (X-Ray) luminosity
luminosity, edd = agn.bh_mass_to_eddington_ratio_luminosity(black_hole_mass, z,
        "Schechter", parameter1=-1, parameter2=-0.65)

# Column density
nh = agn.luminosity_to_nh(luminosity, z)


