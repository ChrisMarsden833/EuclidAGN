# EuclidAGN

Working repository for the code we are working on to add AGN to any mock catalog, but with the express aim of adding it to the Euclid Flagship Mock.

Main functions are held in AGNCatalogToolbox.py. For a simple implementation of these functions, they are called in Simple_Example.ipynb. An object oriented 'wrapper' that makes duplicate testing easier is implemented in ACTTestingEncapsulation.py. The notebook demonstration of this is shown in Encapsulated Example.ipynb.

## Data

Data is included, except for the large scale simulation catalogs, which should be formatted as .npy files, as structured arrays. These are read using the function load_halo_catalog(), within AGNCatalogToolbox.py, and should be stored within the ./BigData/ directory. Within this directory are some resources to help process these files.
