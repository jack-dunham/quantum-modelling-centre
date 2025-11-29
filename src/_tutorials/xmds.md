---
title: XMDS
author:
  - jdunham
  - wbruzda
---

{% assign authors_raw = page.authors | default: page.author %}
{% assign authors = authors_raw | arrayify %}

{% if authors %}
  <p class="page__meta" style="margin-top: 0.5em; margin-bottom: 2.0em; line-height: 1.2; color: grey; font-size: 1.0em; font-style: italic;">
    By
    {% for author_id in authors %}
      {% assign author = site.data.authors[author_id] %}
      {{ author.name }}{% if forloop.last == false %}, {% endif %}
    {% endfor %}
  </p>
{% endif %}


# Installing XMDS locally

XMDS(2) is an open-source package for running numerical simulations and solving systems of partial differential equations (PDEs). It allows the user to write a high-level input file describing the problem and automatically generates optimized C/C++ code to solve it.

This section will walk you through building XMDS as user. This guide has been adapted from [the documentation](http://www.xmds.org/installation.html). XMDS can be a little tricky to install as a user due to the penchant of `pip`'s to install packages system wide. If you do not have appropriate rights to install software system wide, *this will not work*.

## Prerequisites
We require the following C libraries: **FFTW, MPI, MLK** and a **C++ compiler**. 
It is assumed these are available in the module list. 
To load these, do 
``` bash
module load libs/fftw
module load libs/mkl
```
These are commands that talk to an environment‑module system. It is a utility most HPC install to let users switch between many different software stacks without polluting the global system. So, the `module load` syntax only works on systems that provide the module infrastructure. On a regular desktop one would

Loading FFTW will load MPI and a compiler as prerequisites.
### Python 3
We also require Python 3:
```
module load apps/python
```
Make sure the following packages are installed: **numpy, setuptools, lxml, h5py, pyparsing, cheetah3**. 	If any are missing, install using:
```bash
pip3 install <package> --user
```
The `--user` option is required as some packages will attempt to install system wide which will fail due to lack of permissions.

### HDF5
HDF5 is a widely used data model, library, and file format for storing and managing large and complex datasets, commonly used in scientific computing. HDF5 is implemented as a C library, but it may not be pre-installed on your HPC cluster or available in the module list. If it is not available, it can be installed from source.

To install from source:

1. Download the [HDF5 source code](https://www.hdfgroup.org/download-hdf5/), selecting the `tar.gz` file from the official list.
2. Move this file to your home directory using a tool such as `scp`.
3. It is suggested to create a directory called `~/software` (or a similar location) if one does not already exist.
4. Navigate to the directory containing the tarball and extract it using:

```bash
tar -xf hdf5-X.X.X.tar.gz
```

with the Xs replaced with the version number you are installing. 
Now we need to compile HDF5. Move into the extracted directory. Then run 
```
./configure --prefix=$HOME/.local
```
This may take a few seconds. If you install your binaries, libraries, and header files in some other directory then change `--prefix` appropriately. If you do not know what these means, then just use that above. Then do:
```
make
```
which may take a while. Once this completes, run 
```
make install
```
to install HDF5.

This should be all the prerequisites required for installing XMDS.
