---
title: 'PyPO: a Python package for Physical Optics'
tags:
  - Python
  - C++
  - CUDA
  - Reflectors
  - Physical Optics
  - Geometrical Optics
authors:
  - name: Arend Moerman
    orcid: 0000-0002-0475-6134
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Maikel H. Gafaji
    equal-contrib: true
    affiliation: 2
  - name: Kenichi Karatsu
    orcid: 0000-0002-0475-6134
    equal-contrib: False
    affiliation: [1,3]
  - name: Akira Endo
    orcid: 0000-0002-0475-6134
    equal-contrib: False
    affiliation: 1
affiliations:
  - name: Faculty of Electrical Engineering, Mathematics and Computer Science, Delft University of Technology, Mekelweg 4, 2628 CD, Delft, The Netherlands
    index: 1
  - name: The Hague University of Applied Sciences, Johanna Westerdijkplein 75, 2521 EN, The Hague, The Netherlands
    index: 2
  - name: SRON—Netherlands Institute for Space Research, Niels Bohrweg 4, 2333 CA, Leiden, The Netherlands
    index: 3
date: 28 February 2023
bibliography: paper.bib
---

# Summary

`PyPO` is a Python interface for end-to-end simulations of (quasi-)optical reflector systems. Using `PyPO`, it is possible to simulate electromagnetic field propagation between multiple reflector surfaces. As numerical methods, `PyPO` can use a geometrical optics (GO) approach using ray-tracing, or the equivalent surface current approach, belonging to the field of physical optics (PO) [@Balanis:1989]. 
After propagation of electromagnetic fields, `PyPO` can calculate a multitude of efficiencies and other figures of merit, allowing for quantitative analysis of the designed (quasi-)optical systems.

# Statement of need

`PyPO` offers the following functionality:

- Convenient workflow for designing and characterising reflector systems consisting of planar and quadric surfaces. Design and simulation can be done in either a simple Python script or through the built-in graphical user interface (GUI).
- Common beam patterns, such as point sources and Gaussian beams, that can be used as input for GO/PO propagation. Custom beam patterns, for example measured in a lab, can also be used as input.
- Efficient multi-threaded C++ libraries for performing GO and PO calculations. If an Nvidia GPU is present, the calculations can be accelerated even more using CUDA.
- Methods for evaluating common figures of merit such as root-mean-square (RMS) spot values for GO propagation. For PO, commonly used metrics such as spillover, taper, aperture, main beam and cross-polar efficiencies can be calculated.

Commercial software for these ends, such as OpticStudio (Zemax) and GRASP (TICRA), has already been developed, but to our knowledge `PyPO` is the first free open-source package that simulates planar and quadric reflector geometries using both GO and PO. 
Moreover, `PyPO` does not employ approximations often employed by other software packages such as `POPPy` [@Perrin:2012] and Prysm [@Dube:2019]. Rather, PyPO directly solves the radiation integral, allowing for propagation between multiple reflector surfaces.

Currently, `PyPO` is heavily used in the laboratory verification and characterisation of the DESHIMA 2.0 spectrometer [@Taniguchi:2022]. 
Specifically, `PyPO` is used for the following purposes:

- Optimisation of correcting optics using the GO calculations.
- PO propagation of measured beam patterns through the Cassegrain setup of the ASTE [@ASTE] telescope.
- Calculation of far-field beam patterns after the ASTE telescope.
- Evaluation of the instrument efficiencies at the ASTE telescope.

# Availability
`PyPO` can be found on [Github](https://github.com/arend95/PyPO). 
Software documentation and instructions regarding installation, contribution and issue tracking can be found in the [documentation](https://arend95.github.io/PyPO/). 
The package comes with several worked examples illustrating the workflow and features, which can be used as building blocks for new reflector systems.

# Acknowledgements
This work is supported by the European Union (ERC Consolidator Grant No. 101043486 TIFUUN). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

# References
