# BALRoGO

[![pipeline status](https://gitlab.com/eduardo-vitral/balrogo/badges/master/pipeline.svg)](https://gitlab.com/eduardo-vitral/balrogo/-/commits/master)
[![coverage report](https://gitlab.com/eduardo-vitral/balrogo/badges/master/coverage.svg)](https://gitlab.com/eduardo-vitral/balrogo/-/commits/master)
[![pypi](https://img.shields.io/pypi/v/balrogo.svg)](https://pypi.python.org/pypi/balrogo/)
[![python](https://img.shields.io/pypi/pyversions/balrogo.svg)](https://pypi.python.org/pypi/balrogo)
[![license](http://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<!-- markdownlint-disable-next-line no-inline-html -->
<img alt="logo" align="right" src="https://gitlab.com/eduardo-vitral/balrogo/-/raw/master/logo.png" width="20%" />

BALRoGO: Bayesian Astrometric Likelihood Recovery of Galactic Objects.

- Specially developed to handle data from the Gaia space mission.
- Extracts galactic objects such as globular clusters and dwarf galaxies from data contaminated by interlopers.
- Uses a combination of Bayesian and non-Bayesian approaches.
- Provides:
  - Fits of proper motion space.
  - Fits of surface density.
  - Fits of object center.
  - Confidence regions for the color-magnitude diagram and parallaxes.

If something does not work, please [file an issue](https://gitlab.com/eduardo-vitral/balrogo/-/issues).

## Attribution

Please cite [us](https://academic.oup.com/mnras/article-abstract/504/1/1355/6237524?redirectedFrom=fulltext) if you find this code useful in your research. The BibTeX entry for the paper is:

```bibtex
@ARTICLE{Vitral21,
   author = {{Vitral}, Eduardo},
    title = "BALRoGO: Bayesian Astrometric Likelihood Recovery of Galactic Objects - Global properties of over one hundred globular clusters with Gaia EDR3",
  journal = {\mnras},
     year = 2021,
    month = jun,
   volume = {504},
   number = {1},
    pages = {1355-1369},
      doi = {10.1093/mnras/stab947},
   eprint = {2102.04841},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.1355V},
}
```

## Quick overview

BALRoGO has nine modules that perform different tasks:

- ***angle.py*** : This module contains the main functions concerning angular tansformations, sky projections and spherical trigonomtry.
- ***gaia.py*** : This module contains the main functions concerning the handling of the Gaia mission data.
- ***hrd.py*** : This module contains the main functions concerning the color magnitude diagram (CMD). It provides a Kernel Density Estimation (KDE) of the CMD distribution.
- ***marginals.py*** : This module is based on the Python corner package (Copyright 2013-2016 Dan Foreman-Mackey & contributors, The Journal of Open Source Software): https://joss.theoj.org/papers/10.21105/joss.00024
I have done some modifications on it so it allows some new features and so it takes into account some choices as default. I thank Gary Mamon for his good suggestions concerning the plot visualization.
-  ***parallax.py*** : This module contains the main functions concerning parallax information. It provides a kernel density estimation of the distance distribution, as well as a fit of the mode of this distribution.
- ***pm.py*** : This module contains the main functions concerning proper motion data. It provides MCMC and maximum likelihood fits of proper motions data, as well as robust initial guesses for those fits.
- ***position.py*** : This module contains the main functions concerning positional information. It provides MCMC and maximum likelihood fits of surface density, as well as robust initial guesses for the (RA,Dec) center of the source.
- ***mock.py*** : This module handles mock data sets. It converts 3D coordinates to sky coordinates and is able to add realistic errors to proper motions. It is also able to generate Milky Way interlopers.
- ***dynamics.py*** : This module handles conversions from celestial coordinates to plane of sky coordinates (vPOSr,vPOSt), as well as allows computation of dispersion profiles in different ways. Also computes the velocity anisotropy from cartesian data.

## Installation

BALRoGO is available through [pip](https://pypi.org/project/balrogo/). The quickiest way to install it is to type the following command in your terminal:

```terminal
pip install balrogo
```

If you are using [Anaconda](https://www.anaconda.com/), you might want to install it directly in your Anaconda bin path:

```terminal
cd path/anaconda3/bin/
pip install balrogo
```

For updated versions of the code, you can do the same as above, but instead of using `pip install balrogo`, you should type:

```terminal
pip install --upgrade balrogo
```

### Using BALRoGO on [*Gaia*](https://www.cosmos.esa.int/web/gaia/data-access) data

For quick tutorial of BALRoGO applied to *Gaia* data, please click [here](https://gitlab.com/eduardo-vitral/balrogo/-/blob/master/GAIA.md).

## License

Copyright (c) 2020 Eduardo Vitral & Alexandre Macedo.

BALRoGO is free software made available under the [MIT License](LICENSE). The BALRoGO logo is licensed under a [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
