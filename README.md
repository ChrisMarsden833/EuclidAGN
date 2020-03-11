# EuclidAGN

Working repository for the code we are working on to add AGN to any mock catalog, but with the express aim of adding it to the Euclid Flagship Mock.

## But how does it work?

The code is encapsulated in a module within the directory AGNCatalogToolbox/. At it's most basic, it is just a series of functions that are in the python file **main.py**. A simple python file in the main directory might call them using **from AGNCatalogToolbox import main** (please look at the docstrings for documentation of each function).

**A really simple implementation of these functions can be seen in simple.py**. A simple semi-analytic catalog is created.

**Please note that we are still working on the best parameters to use for these functons. In general, the default values are pretty good, but please ask us if you are unsure**

To make testing easier, I've written an object oriented testing framework, called TestingFramework.py. You can see this working in the notebook **Fitting at z = 0**. I reccomend this if you are working on the code itself, as it allows us to run may iterations without repeated code. The redshift is entirely dynamic, in the sense that you can just change the z value at the top of the notebook and everything should work, including selecting the best files available, etc.

## Data

Data is included, **except for the large scale simulation catalogs** if you want to do clustering etc. These should be formatted as .npy files, as structured arrays (for speed). These are read using the function load_halo_catalog(), within main.py, and should be stored within the ./BigData/ directory. Within this directory are some resources to help process these files. You will need ~100GB of storage per file to process these from scratch from the cosmosim website.

**For now, you will need the Data/ directory wherever you choose to import this library, so I suggest you just write your code in the main directory**
