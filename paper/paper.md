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

Physical optics (PO) is a high-frequency approximation commonly used for solving electromagnetic scattering problems [@Balanis:1989]. 
In this approximation, the electromagnetic field scattered by an object is obtained by calculating electric and magnetic surface currents induced by an incoming electromagnetic field.
These currents are calculated using geometrical optics (GO) by assuming that, locally, the field impinging on the object is a plane wave. Using Snell's law, the reflected field can be calculated and, together with the impinging field, the induced surface currents can be calculated. 
The calculated current distributions can then be used to calculate the 

This method has been used frequently in fields such as radar engineering and telescope design. It has proven to be an accurate method for high frequency problems, i.e. problems where the dimensions of the scattering objects are substantially larger than the wavelength under consideration.

# Statement of need

`PyPO` is a Python interface for end-to-end simulations of general reflector systems using GO/PO.
It offers the following functionality:
 * Convenient workflow for designing and characterising reflector systems consisting of planar and quadric surfaces. Design and simulation can be done in either a simple Python script or through the built-in graphical user interface (GUI).
 * Common beam patterns, such as point sources and Gaussian beams, that can be used as input for GO/PO propagation. Custom beam patterns, for example measured in a lab, can also be used as input.
 * Efficient C++/CUDA libraries for performing GO and PO calculations.
 * Methods for evaluating common figures of merit such as root-mean-square (RMS) spot values for GO propagation. For PO, commonly used metrics such as spillover, taper, aperture, main beam and cross-polar efficiencies can be calculated.

Commercial software for these ends, such as Zemax and GRASP, has already been developed, but to our knowledge `PyPO` is the first free open-source package that simulates planar and quadric reflector geometries using both GO and PO. 
Moreover, `PyPO` does not resort to approximations such as the Fresnel and Fraunhofer approximation commonly used in other software packages. Rather, PyPO directly solves the radiation integral, extending the simulation possibilities to cases where the previously mentioned approximations might fail.

Currently, `PyPO` is heavily used in the laboratory verification and characterisation of the DESHIMA 2.0 spectrometer [@Taniguchi:2022].

# Availability
`PyPO` can be found on [Github](https://github.com/arend95/PyPO). Instructions regarding installation and documentation can be found in the [documentation](https://arend95.github.io/PyPO/). The package comes with several examples illustrating the workflow and features, which can be used as building blocks for new reflector systems.

# Acknowledgements
This work is supported by the European Union (ERC Consolidator Grant No. 101043486 TIFUUN). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

# References