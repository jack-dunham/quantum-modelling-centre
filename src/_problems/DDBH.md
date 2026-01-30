---
title: "Driven Dissipative Bose-Hubbard Model"
collection: problems
author: aferrier
show_author: true
toc_sidebar: true
layout: single
classes: wide
excerpt: "In this tutorial, we will study the behaviour of nonlinear bosonic lattice models with drive and decay."
---

{% if page.author %}
  {% assign author_id = page.author %}
  {% assign author = site.data.authors[author_id] %}
  <p class="page__meta" style="margin-top: 0.5em; margin-bottom: 2.0em; line-height: 1.2; color: grey; font-size: 1.0em; font-style: italic;">
    By {{ author.name }}
  </p>
{% endif %}

In this tutorial, we will use [Positive P]({{ site.baseurl }}/software/DDBHPP) to study the behaviour of some examples of driven dissipative Bose-Hubbard models.

# Recommended tutorials

[xmds2]({{ site.baseurl }}/tutorials/xmds)

# Introduction

In the most general form, without at this stage specifying the lattice geometry, the Hamiltonian for a driven Bose-Hubbard model can be written in the frame rotating with the driving frequency as

\begin{equation}
\hat{H} = \sum_j \left( -\Delta\hat{a}^\dagger_j\hat{a}_j +\frac{U}{2}\hat{a}^\dagger_j\hat{a}^\dagger_j\hat{a}_j\hat{a}_j +F_j\hat{a}^\dagger_j + F_j^*\hat{a}_j \right) -\sum_{j,j'} \left(J_{j,j'}\hat{a}^\dagger_j\hat{a}_{j'} + J^*_{j,j'}\hat{a}^\dagger_{j'}\hat{a}_j \right) \, ,
\end{equation}

where $\Delta$ gives the detuning between the on-site energy and the driving frequency, $U$ the two-body interaction strength, $F_j$ the driving amplitude on each site $j$, and $J_{j,j'}$ the hopping between sites (typically $J_{j,j'} = J$ for connected sites and 0 otherwise).  Including the effects of dissipation to the external environment, the evolution of the system as described by the many-body density matrix $\hat{\rho}$ is given by a Markovian Lindblad master equation

\begin{equation} 
    \frac{\partial\hat{\rho}}{\partial t} = -i\left[\hat{H}, \hat{\rho}\right] + \sum_j\frac{\gamma}{2}\left(2\hat{a}_j\hat{\rho}\hat{a}^\dagger_j - \hat{a}^\dagger_j\hat{a}_j\hat{\rho} - \hat{\rho}\hat{a}^\dagger_j\hat{a}_j\right) 
\end{equation}

where $\gamma$ is the dissipation rate (we set $\gamma=1$ in our units).












