---
title: "DDBHPP"
author: aferrier
header: 
    teaser: /assets/images/key_image_ppdd.png
github: "A-Ferrier/DDBHPP"
layout: single
classes: wide
excerpt: "Positive-P simulations for Driven-dissipative Bose Hubbard models written for xmds2, with optional Matlab post-processing library. Keywords: Open Quantum Systems, Dynamics, Stochastic Phase space methods, Positive P, Bosons, Lattices"
---

{% if page.author %}
  {% assign author_id = page.author %}
  {% assign author = site.data.authors[author_id] %}
  <p class="page__meta" style="margin-top: 0.5em; margin-bottom: 2.0em; line-height: 1.2; color: grey; font-size: 1.0em; font-style: italic;">
    By {{ author.name }}
  </p>
{% endif %}

Positive-P simulations for Driven-dissipative Bose Hubbard models written for xmds2, with optional Matlab post-processing library. 

Keywords: Open Quantum Systems, Dynamics, Stochastic Phase space methods, Positive P, Bosons, Lattices

# Description:

A framework for generating stochastic trajectories in the positive-P representation for solving driven dissipative Bose-Hubbard models using the differential equation solving package [xmds2](http://www.xmds.org/).  A library of post-processing codes written for Matlab for calculating some quantum observables from corresponding averages of the stochastic trajectories is also provided.  

# Installation:

## Using the code for simulations 
Once you have xmds2 installed, make a copy of the template directory for the geometry you wish to simulate, e.g. DDBH_Nsite_PP_template for 1D chains.  Use the command 
``` bash
xmds2 DDBH_Nsite_PP.xmds
```
to compile the xmds script and generate an executable.  The executable generates a single trajectory when run; the template directory also contains a bash script (DDBH_Nsite_PP_runscript.sh) to loop running for the desired numbers of stochastic trajectories and rename the output files appropriately.  Edit this runscript to set the system parameters and number of trajectories as desired, and then use 
``` bash
./DDBH_Nsite_PP_runscript.sh
```
 to have the simulations run to generate positive-P trajectories.  

## Matlab tools for calculating observables
The directory matlab_tools contains a library of Matlab scripts for loading the output data into Matlab and calculating observables from the corresponding averages.  To use, make a copy of the matlab_tools directory; then, in the simulation directory, edit the file analysis.m so that the parameters match those used for the simulation, and that the string toolspath points to the correct location for your matlab_tools directory.  Add the names of any of the scripts from matlab_tools you wish to include in the analysis to the list at the bottom of analysis.m.  Running analysis.m in Matlab from within the simulation directory will then calculate the observables and save the results to the output file whose name is defined in analysis.m.  To run the analysis on a system with Matlab installed without opening a Matlab window you can use the command
``` bash
matlab -nosplash -nodesktop -nodisplay < analysis.m
```
from inside the simulation directory.  

# Example Problems:

[Driven Dissipative Bose-Hubbard models]({{ site.baseurl }}/problems/DDBH)

# Further reading:
[P. Deuar, A. Ferrier, M. Matuszewski, G. Orso, M. H. Szymańska, Fully Quantum Scalable Description of Driven-Dissipative Lattice Models. PRX Quantum 2, 010319 (2021)](https://doi.org/10.1103/PRXQuantum.2.010319)

[A. Ferrier, M. Matuszewski, P. Deuar, M. H. Szymańska, Antibunching in locally driven dissipative Lieb lattices. arXiv:2512.01645 [quant-ph].](https://arxiv.org/abs/2512.01645)
