---
title: "Driven Dissipative Bose-Hubbard Model"
collection: problems
author: A-Ferrier
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

In this tutorial, we will use exact diagonalization (in a truncated Fock basis), and [Positive P]({{ site.baseurl }}/software/DDBHPP) to study the behaviour of some examples of driven dissipative Bose-Hubbard models.

# Recommended tutorials

[xmds2]({{ site.baseurl }}/tutorials/xmds)

# Introduction

We consider a transverse-field Ising model in one-dimension described by the Hamiltonian
\begin{equation}
H = J\sum_i \sigma^z_i \sigma^z_{i+1} + h\sum_i \sigma^x_i.
\end{equation}
The ground state properties of the system are well understood.
Here, we are interested in exploring its non-equilibrium properties. We assume that the interactions of the system of spins with the external enviornment is well described by a Markovian Lindblad master equation for the many-body density matrix $\rho$,
\begin{equation} 
    \partial_t \rho = -i[H,\rho] + \sum_k \left(\Gamma_k \rho \Gamma_k^\dagger - \frac{1}{2}\{\Gamma_k^\dagger \Gamma_k,\rho\}\right) := \mathcal{L}[\rho]
\end{equation}
The superoperator $\mathcal{L}$ is often referred to as the Lindbladian. Here, each jump operator $\Gamma_k$ encodes a specific dissipative process acting on site $k$. In this example we choose
\begin{equation} 
\Gamma_k = \sqrt{\gamma}\,\sigma^-_k,
\end{equation}
where
- $\sigma^- = (\sigma^x - i\sigma^y)/2$ is the spin-lowering operator on a single site.
- $\gamma$ is the dissipation rate (we set $\gamma=1$ in our units).

Physically, $\Gamma_k$ implements spontaneous emission of an excited spin into the environment, driving each spin toward the $|\downarrow\rangle$ state.  The competition among the interaction between neighbouring spins ($J\sigma^z_i\sigma^z_{i+1}$), coherent transverse-field flips ($h\sigma^x$) and dissipative decay ($\sqrt{\gamma}\sigma^-$) gives rise to nontrivial non-equilibrium dynamics and steady states.  Other common choices include:
- Dephasing: $\Gamma_k = \sqrt{\gamma_z}\sigma^z_k$, which randomizes the relative phase in the $[\ket{\uparrow}, \ket{\downarrow}]$ basis without changing populations.
- Collective decay: $\Gamma = \sqrt{\gamma_c},\sum_k \sigma^-_k$, coupling the entire chain to a common bath and leading to superradiant effects.








# Two-dimensional lattices - iPEPO

To be completed.
