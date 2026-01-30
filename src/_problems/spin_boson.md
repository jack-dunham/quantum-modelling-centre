---
title: "Spin-Boson Model"
collection: problems
author: aferrier
show_author: true
toc_sidebar: true
layout: single
classes: wide
excerpt: "In this tutorial, we will study the behaviour of coupled (Jaynes-Cummings) spin-boson models with drive and decay."
---

{% if page.author %}
  {% assign author_id = page.author %}
  {% assign author = site.data.authors[author_id] %}
  <p class="page__meta" style="margin-top: 0.5em; margin-bottom: 2.0em; line-height: 1.2; color: grey; font-size: 1.0em; font-style: italic;">
    By {{ author.name }}
  </p>
{% endif %}

In this tutorial, we will use [Positive P]({{ site.baseurl }}/software/SBPP) to study the behaviour of some examples of driven dissipative Jaynes-Cummings models and Jaynes-Cummings-Hubbard lattice models.  

# Recommended tutorials

[xmds2]({{ site.baseurl }}/tutorials/xmds)

# Introduction

While there are many different models for coupled systems of spins and bosons depending on the exact physical arrangement and approximations being used, here we will focus on the Jaynes-Cummings interaction, where each qubit (Spin operators $\hat{S}^{X,Y,Z}_j = \frac{1}{2}\sigma^{X,Y,Z}_j$, $\hat{S}^\pm_j = \hat{S}^X_j \pm i\hat{S}^Y_j$, with $\sigma^{X,Y,Z}_j$ the usual $2\times2$ Pauli matrices) is coupled to a local bosonic mode with annihilation operator $\hat{a}_j$.  The local Hamiltonian for each site is given by

\begin{equation}
\hat{H}^{JC}_j = \omega_C\hat{a}^\dagger_j\hat{a}_j +\omega_0\hat{S}_j^Z +g\left(\hat{a}_j\hat{S}_j^+ +\hat{a}^\dagger_j\hat{S}_j^-\right) +h\hat{S}_j^X +F\left(\hat{a}_j+\hat{a}^\dagger_j\right)\, ,
\end{equation}

with $\omega_C$ the boson energy, $\omega_0$ the qubit energy, $g$ the the strength of the Jaynes-Cummings interaction, and we have included coherent driving on both the spins $h$ and bosons $F$.  For the case of many-site Jaynes-Cummings-Hubbard lattices, connected sites $(j,j')$ are coupled by a bosonic hopping $J_{j,j'}$, leading to a total Hamiltonian

\begin{equation}
\hat{H} = \sum_j \hat{H}^{JC}_j -\sum_{j,j'} \left(J_{j,j'}\hat{a}^\dagger_j\hat{a}_{j'} + J^*_{j,j'}\hat{a}^\dagger_{j'}\hat{a}_j \right) \, .
\end{equation}

Including dissipation, the state of the system evolves according to the master equation

\begin{equation} 
    \frac{\partial\hat{\rho}}{\partial t} = -i\left[\hat{H}, \hat{\rho}\right] + \sum_j \left(\frac{\gamma}{2}D[\hat{a}_j] + \frac{\Gamma_\downarrow}{2}D[\hat{S}^-_j] + \frac{\Gamma_\phi}{2}D[\hat{S}^Z_j] \right)\, ,
\end{equation}

where $D[\hat{X}] = 2\hat{X}\hat{\rho}\hat{X}^\dagger - \hat{X}^\dagger\hat{X}\hat{\rho} - \hat{\rho}\hat{X}^\dagger\hat{X}$ are Lindbladian dissipation terms, for which we consider three kinds of dissipation: single particle bosonic dissipation $\gamma$, spin decay $\Gamma_\downarrow$, and spin dephasing $\Gamma_\phi$.  

# Single site

To be completed.

# Jaynes-Cummings-Hubbard lattices

To be completed.  
