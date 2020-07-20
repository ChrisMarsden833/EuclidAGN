# EuclidAGN, or the "AGN Catalog Toolbox"

This is a repository for the "AGN Catalog Toolbox", a project with the express aim of adding Active Galaxies (Active Galactic Nuclei, AGN) to any mock galaxy catalog. This project was developed initially for Euclid Flagship Mock, but we are hoping to use it for other projects.

Primary contributers are:

1. Christopher Marsden (c.marsden@soton.ac.uk), PhD student at the University of Southampton
2. Dr Viola Allevato (viola.allevato@sns.it), Marie Curie Fellow at Scuola Normale Superiore
3. Dr Francesco Shankar (f.shankar@soton.ac.uk), Associate Professor, University of Southampton

## How does it work?

We aim to assign AGN to (in principle) and mock catalog, based on empirical relations but built from the "bottom up", starting with the dark matter halo, assigning supermassive black hole mass and AGN luminosity.

## But **how** does it work?

The code is encapsulated in a module within the directory AGNCatalogToolbox/. At it's most basic, it is just a series of separate functions that are in the python file **main.py**. 

**A really simple implementation of these functions can be seen in simple.py**. A simple semi-analytic catalog is created. **Please note that we are still working on the best parameters to use for these functons. In general, the default values are pretty good, but please ask us if you are unsure**.

To make testing easier, I've written an object oriented testing framework, called TestingFramework.py. You can see this working in the notebook **Fitting at z = 0**. I recommend this if you are working on the code itself, as it allows us to run may iterations without repeated code.

## Data

Data is included, **except for the large scale simulation catalogs** if you need 3D coordinates (e.g. for clustering). If you want these, you need to construct them, which can be done as follows.

1. Download the relevant simulation data from the cosmosim website. The exact format of this data will depend on the simulation you want (and how you download it), but I reccomend the catalog/snapshot files, and download them using wget.

2. These should be formatted as .npy files, as structured arrays (for speed). These are read using the function load_halo_catalog(), within main.py, and should be stored within the ./BigData/ directory. Within this directory are some resources to help process these files. You will need ~100GB of storage per file to process these from scratch from the cosmosim website, but the processed .npy files are smaller (i.e. do this on a server).

**For now, you will also need the Data/ directory for numerical relations used by the code in the same path as your script**

## Prerequisites

'Uncommon' Python packages:

* colossus (pip install colossus)
* Corrfunc (pip install Corrfunc - requires GSL)
* numba
