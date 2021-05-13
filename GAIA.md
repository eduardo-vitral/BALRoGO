# Handling ***Gaia*** data

BALRoGO was develop with *Gaia* data in mind, so this document depicts an example with data from the Sculptor dwarf spheroidal ([PGC 3589](https://en.wikipedia.org/wiki/Sculptor_Dwarf_Galaxy)).

By the end of this tutorial, you should be able to:

- [ ] Download data from the Gaia archive.
- [ ] Install and update BALRoGO.
- [ ] Run BALRoGO in a Python console.
- [ ] Change default parameters in BALRoGO and adapt them to your analyzed source.

## Download the data

The first step is to acquire the data that BALRoGO will fit. We encourage you to download it directly from the *Gaia* archive [here](https://gea.esac.esa.int/archive/).

1. Click on "Search", and then type the name PGC 3589 (or other source you want to fit) in the box "Name".
2. Then, click on "Display columns" and select all entries.
3. Click on "Show query", at the bottom of the page.

You are now directed to the Advanced query page. The following lines should appear on the query box:

```javascript
SELECT TOP 500 *
FROM gaiaedr3.gaia_source 
WHERE 
CONTAINS(
	POINT('ICRS',gaiaedr3.gaia_source.ra,gaiaedr3.gaia_source.dec),
	CIRCLE(
		'ICRS',
		COORD1(EPOCH_PROP_POS(15.03916667,-33.70888889,0,.0900,.0200,111.4000,2000,2016.0)),
		COORD2(EPOCH_PROP_POS(15.03916667,-33.70888889,0,.0900,.0200,111.4000,2000,2016.0)),
		0.001388888888888889)
)=1
```

We are going to select all sources inside a two degrees cone. For that, remove the `'TOP 500'` command (for selecting all stars) and change `'0.001388888888888889'` to `'2.0'` (two degrees):

```javascript
SELECT *
FROM gaiaedr3.gaia_source 
WHERE 
CONTAINS(
	POINT('ICRS',gaiaedr3.gaia_source.ra,gaiaedr3.gaia_source.dec),
	CIRCLE(
		'ICRS',
		COORD1(EPOCH_PROP_POS(15.03916667,-33.70888889,0,.0900,.0200,111.4000,2000,2016.0)),
		COORD2(EPOCH_PROP_POS(15.03916667,-33.70888889,0,.0900,.0200,111.4000,2000,2016.0)),
		2.0)
)=1
```

Note that you can change the *Gaia* release you want to analyze by changing the line `'FROM gaiaedr3.gaia_source'`.

Submit your query and download the file in **FITS** format.

- [x] Download data from the Gaia archive.

## Running BALRoGO

### Installation

Once you have your data `SCULPTOR.fits` in your `path`, you need to make sure that you have BALRoGO installed and updated. For that, check the section **Installation** in the [README](https://gitlab.com/eduardo-vitral/balrogo/-/blob/master/README.md) file.

- [x] Install and update BALRoGO.

### Python script

Congratulations, you have BALRoGO installed and you have the data to use it, so from now on, you can model globular clusters, dwarf galaxies and mock data as you wish. Below, we show an example of script for the Sculptor Dwarf Spheroidal.

The function `extract_object` from the `gaia` method is supposed to clean the data and fit the proper motion and surface density of your source, finally providing the stars that have high probability of belonging to the analyzed object. The input you must provide is the path to your .FITS file:

```python
from balrogo import gaia

# Read HDU
print("Reading input...")
path = r"path/SCULPTOR.fits"

# Run sample code
print("Running code...")
final_data, results_sd, var_sd, results_pm, var_pm = gaia.extract_object(path)

print("results_pm:", results_pm)
print("var_pm:", var_pm)

print("results_sd:", results_sd)
print("var_sd:", var_sd)
```

- [x] Run BALRoGO in a Python console.

## Optional parameters

BALRoGO is usually able to perform its routine over most of *Gaia* sources with its default parameters. However, we strongly encourage the user to adapt them to the source in question. Here is a small description of the main optional parameters you can change in the function `extract_object`:

- ***sd_model*** : string, optional
    Surface density model to be considered. Available options are:
    - `'sersic'`
    - `'kazantzidis'`
    - `'plummer'`
    - `'gplummer'`
    - `'test'`, for testing which among Plummer, Kazantzidis and Sersic should be used, based on AICc.
    The default is `'plummer'`.
- ***prob_method*** : string, optional
    Method of probability cut chosen. Available options are:
    - `'complete'`: Considers proper motions and projected radii.
    - `'pm'`: Considers only proper motions.
    - `'position'`: Considers only projected radii.
    The default is `'complete'`.
- ***prob_limit*** : float, optional
    Probability threshold (between 0 and 1) considered to cut the data. The default is `0.9`.
- ***use_hrd*** : boolean, optional
    True if the user wants to select stars inside a n-sigma contour region inside the color-magnitude diagram.
    The default is `True`.
- ***nsig*** : float, optional
    Number of sigma up to when consider stars inside the KDE color-magnitude diagram.
    The default is `3.3`.
- ***bw_hrd*** : float or string, optional
    Bandwidth method of KDE's scipy method. The default is `None`.
- ***r_max*** : float, optional
    Maximum projected radius up to where consider the data.
    The default is `None`.
- ***err_lim*** : float, optional
    Error threshold for proper motions. The default is `None`.
- ***err_handle*** : string, optional
    Way in which the error limit is handled:
    - `'relative'`, if the err_lim argument is the number of times the initial guess on the galactic object proper motion dispersion.
    - `'absolute'`, if the err_lim argument is the absolute error limit in mas/yr.
    The default is `'relative'`.
- ***object_type*** : string, optional
    Galactic object type, to be considered when defining the default value of err_lim:
    - `'gc'`, for globular cluster.
    - `'dsph'`, for dwarf spheroidal galaxy.
    The default is `'gc'`.
- ***return_center*** : boolean, optional
    True if the user wants an estimate of the center as output.
    The default is `False`.
- ***check_fit*** : boolean, optional
    True is the user wants to plot validity checks throughout the fitting procedure. The default is `False`.

The reader is also encouraged to use the other methods of BALRoGO, specially, ***pm.py***, ***position.py***, ***mock.py*** and ***angle.py***.

- [x] Change default parameters in BALRoGO and adapt them to your analyzed source.
