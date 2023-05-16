---
title: 'PyPO: a Python package for Physical Optics'
tags:
  - Physical Optics
  - Geometrical Optics
  - Reflectors
  - Python
  - C++
  - CUDA
authors:
  - name: Arend Moerman
    orcid: 0000-0002-0475-6134
    corresponding: true
    affiliation: 1
  - name: Maikel H. Gafaji
    affiliation: 2
  - name: Kenichi Karatsu
    orcid: 0000-0003-4562-5584
    affiliation: [1,3]
  - name: Akira Endo
    orcid: 0000-0003-0379-2341
    affiliation: 1
affiliations:
  - name: Faculty of Electrical Engineering, Mathematics and Computer Science, Delft University of Technology, Mekelweg 4, 2628 CD, Delft, The Netherlands
    index: 1
  - name: The Hague University of Applied Sciences, Johanna Westerdijkplein 75, 2521 EN, The Hague, The Netherlands
    index: 2
  - name: SRON—Netherlands Institute for Space Research, Niels Bohrweg 4, 2333 CA, Leiden, The Netherlands
    index: 3
date: 26 April 2023
bibliography: paper.bib
---

# Summary
`PyPO` is a Python interface for end-to-end design, simulation and analysis of (quasi-)optical reflector systems. 
It can model the forward and backward propagation of electromagnetic field distributions between multiple planar and (off-axis) quadric surfaces, as well as far-field propagation.
Simulations are performed using either geometrical optics (GO) or the equivalent surface current approach, belonging to the field of physical optics (PO) [@Balanis:1989].
The GO and PO calculations are performed using libraries written in C++ and CUDA, allowing for multi-threading and GPU acceleration.
Common figures of merit, such as aperture efficiency and half-power beamwidth, can be calculated and used for quantitative analysis of the designed system.
Input beam patterns can be selected from a range of models, such as Gaussian beams, point sources and uniform current distributions. 
Custom beam patterns can also be imported to, for example, model the propagation of measured beam patterns through simulated optical systems.

`PyPO` can be used through either a scripting-based approach, where simulations are defined in Python scripts, or through the graphical user interface (GUI).
It only carries core dependencies on NumPy [@Harris:2020], Matplotlib [@Hunter:2007] and SciPy [@Virtanen:2020]. The unittesting framework carries a dependency on nose2. The GUI carries dependencies on PyQt5 and attrs.

# Statement of need
Development of `PyPO` started with the need for alignment strategies for the wideband sub-mm spectrometer DESHIMA 2.0 [@Taniguchi:2022]. 
A software package capable of efficient GO and PO calulations through optical systems consisting of off-axis quadric surfaces was necessary for calculating the configuration of the corrective optics. 
Currently, `PyPO` is also being used in simulations of measured beam patterns of DESHIMA 2.0 at the ASTE [@ASTE] telescope for the analysis of instrument and optical system performance.

Commercial software for GO and PO calculations and analysis, such as OpticStudio (Zemax) and GRASP (TICRA), has already been developed, but to our knowledge `PyPO` is the first free open-source Python package that simulates planar and quadric (off-axis) reflector geometries using both GO and PO. 
Moreover, `PyPO` does not employ approximations often employed by other software packages such as `POPPy` [@Perrin:2012] and Prysm [@Dube:2019]. 
Rather, PyPO directly solves the radiation integral, allowing for propagation between multiple (off-axis) reflector surfaces.

# Availability
`PyPO` can be found on [Github](https://github.com/PyPO-dev/PyPO) and is available for Linux, MacOS and Windows.
Software documentation and instructions regarding installation, contributing and issue tracking can be found in the [documentation](https://pypo-dev.github.io/PyPO-docs/).
The package comes with several Jupyter Notebook tutorials illustrating the workflow and features, and can be used as building blocks for new reflector systems.
Several tutorials for using the GUI features are also included and can be found in the software documentation.

# Acknowledgements
We thank Shahab Oddin Dabironezare for useful discussions and proofreading the manuscript.
This work is supported by the European Union (ERC Consolidator Grant No. 101043486 TIFUUN). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

# References
