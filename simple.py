from AGNCatalogToolbox import main as agn
from colossus.cosmology import cosmology


cosmo = 'planck18'
cosmology = cosmology.setCosmology(cosmo)
volume = 200**3

z = 0

halos = agn.generate_semi_analytic_halo_catalogue(volume, (12, 16, 0.1), z, 0.7)
stellar_mass = agn.halo_mass_to_stellar_mass(halos, z)
black_hole_mass = agn.stellar_mass_to_black_hole_mass(stellar_mass, method="Shankar16", scatter="Intrinsic")

duty_cycle = agn.to_duty_cycle("Geo", stellar_mass, black_hole_mass, z)

luminosity = agn.black_hole_mass_to_luminosity(black_hole_mass, duty_cycle, stellar_mass, z)

nh = agn.luminosity_to_nh(luminosity, z)
agn_type = agn.nh_to_type(nh)

