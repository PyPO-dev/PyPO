---
title: 'PyPO: a Python package for Physical Optics'
tags:
  - Python
  - C/C++
  - CUDA
  - Reflectors
  - Physical Optics
  - Geometrical Optics
authors:
  - name: Arend Moerman
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Maikel H. Gafaji
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Faculty of Electrical Engineering, Mathematics and Computer Science, Delft University of Technology, Mekelweg 4, 2628 CD,
Delft, The Netherlands
   index: 1
 - name: The Hague university of applied sciences, Johanna Westerdijkplein 75, 2521 EN, The Hague, The Netherlands
   index: 2
date: 13 August 2017
bibliography: paper.bib

---

# Summary

Physical optics (PO) is a high-frequency approximation commonly used for solving
electromagnetic scattering problems [@Balanis89]. In this approximation, the electromagnetic field scattered by 
an object is obtained by first integrating the total field illuminating the object. 
Then, using geometrical optics (GO), the induced electromagnetic currents are calculated. 
These currents are then used to calculate the resultant scattered field and can be either analysed or
propagated to the next object.

This method has been used frequently in fields such as radar engineering and telescope design. For example, 

# Statement of need

`PyPO` is a Python interface for end-to-end simulations of general reflector systems using GO/PO.
It offers the following functionality:
* Convenient workflow for designing and characterising reflector systems consisting of planar and quadric surfaces. Design and simulation can be done in either a simple Python script or through the built-in graphical user interface (GUI).
* Common beam patterns that can be used as input for GO/PO propagation. Custom beam patterns, for example measured in a lab, can also be used as input.
* Efficient C/C++/CUDA libraries for performing GO and PO calculations, combining the performance of these compiled languages with the ease-of-use offered by Python.
* Methods for evaluating common figures of merit such as root-mean-square (RMS) spot values for GO propagation. For PO, efficiencies such as spillover, taper, aperture, main beam and cross-polar can be calculated.

Commercial software, such as Zemax and GRASP, for these ends has already been developed, but to our knowledge `PyPO` is the first free open-source package that simulates planar and quadric reflectors using both GO and PO. 

# Availability
`PyPO` can be found on [Github](https://github.com/arend95/PyPO). Instructions regarding installation and low-level descriptions of the source code can be found in the [documentation](https://arend95.github.io/PyPO/). The package comes with several examples illustrating the workflow and features, and can be used as building blocks for new reflector systems.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
