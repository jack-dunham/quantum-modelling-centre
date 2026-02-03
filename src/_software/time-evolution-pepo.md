---
title: "TimeEvolutionPEPO.jl"
author: jdunham
header:
    teaser: /assets/images/time-evolution-pepo-new.png
    caption: "From:"
github: "jack-dunham/TimeEvolutionPEPO.jl"
deps:
    - name: julia
      version: 1.11
layout: single
classes: wide
docs: "https://jack-dunham.github.io/TimeEvolutionPEPO.jl/dev/"
software_links: true
excerpt: "TimeEvolutionPEPO.jl is a high-level and domain-specific package for simulating the time-evolution of an open-quantum system represented by the iPEPO ansatz. A variety of options are implemented under one common interface for both Lindblad real-time evolution and thermal state annealing."
---

{% if page.author %}
  {% assign author_id = page.author %}
  {% assign author = site.data.authors[author_id] %}
  <p class="page__meta" style="margin-top: 0.5em; margin-bottom: 2.0em; line-height: 1.2; color: grey; font-size: 1.0em; font-style: italic;">
    By {{ author.name }}
  </p>
{% endif %}

TimeEvolutionPEPO.jl is a high-level and domain-specific package for simulating the time-evolution of an open-quantum system represented by the iPEPO ansatz. A variety of options are implemented under one common interface for both Lindblad real-time evolution and thermal state annealing.

Keywords: Open Quantum Systems, Tensor Networks, Spins


# Installation

To use this package, first create a new directory (or navigate to an existing directory) and start Julia from the command line:
```bash
mkdir MyProject
cd MyProject
julia
```
You should then activate an enviroment withing the directory to avoid polluting your global enviroment.
First open the package manager interface by typing `]` at the Julia prompt:
```julia
julia>]
```
Then from the package interface, 
```julia
(@v1.11) pkg> activate .
```
which activates (or creates) an enviroment in current directory, indiciated by the `.`.
Packages can then be added to this enviroment:
```julia
(MyProject) pkg> add TimeEvolutionPEPO
```
which will install the TimeEvolutionPEPO package, and add it to the enviroment. 
Note the change in prompt from `(@v1.11)` (the global enviroment) to `(MyProject)` (the local enviroment).
Additional packages that might be useful can also be added:
```julia
(MyProject) pkg> add Plots, DataFrames
```

# Code Example

```julia
function thermalising(; Jz = 1.0, D = 2)
    _, _, Z = PAULI

    # Spin-1/2 local Hilbert space dimension
    localdim = 2

    # Initialise to the infinite temperate thermal state on 2x2 unit cell lattice
    rho = PEPO(fill(ThermalState(),2,2), localdim, D)

    # Define the square lattice Ising model critical temperature
    βc = log(1 + sqrt(2)) / 2

    # Construct the anti-ferromagnetic Ising model
    model = Model(Jz * LocalOp(Z, Z))

    # Define a method, here we use time-evolving block decimation (TEBD) with default options
    method = TEBD()

    # We also need to define how we compute observables. Here we use the VUMPS algorithm
    obsalg = VUMPS(bonddim=16, maxiter=200, tol=1e-8)

    # We then set up the problem we wish to solve
    sim = Simulation(rho, model, method; timestep = 0.002 * βc)

    # Define an empty vector to store our output
    xis = Float64[]

    # Run the simulation!
    simulate!(sim; numsteps = 1000, maxshots=100) do simstep
        if simulationinfo(simstep).iterations == 0
            return nothing
        end

        # Compute the density matrix
        dm = DensityMatrix(quantumstate(simstep), obsalg)

        # Append the correlation length to the output vector
        push!(xis, correlationlength(dm))

        return nothing
    end

    return xis
end
```

# Related Literature 


