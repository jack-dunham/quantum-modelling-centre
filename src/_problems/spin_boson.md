---
title: "Spin-Boson Model"
collection: problems
author: A-Ferrier
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

In this tutorial, we will use exact diagonalization (in a truncated Fock basis for bosons), and [Positive P]({{ site.baseurl }}/software/SBPP) to study the behaviour of some examples of driven dissipative Jaynes-Cummings models and Jaynes-Cummings-Hubbard lattice models.  

# Recommended tutorials

[xmds2]({{ site.baseurl }}/tutorials/xmds)
