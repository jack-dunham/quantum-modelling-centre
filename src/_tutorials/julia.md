---
title: "Julia"
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

[Julia](https://julialang.org) is an open-source programming language particularly suited to research software engineering.
In this guide, we provide you with some *opinionated* tips on using the language.
Please also consult the [Further Reading](#further-reading) section for other useful information from the community.

# Installation

To install Julia, you should download and install the official Julia version manager and multiplexer [`juliaup`](https://github.com/JuliaLang/juliaup) by running
```bash
curl -fsSL https://install.julialang.org | sh
```
This should also install the latest version of Julia.

# Best Practices

## Package and environment management

Julia ships with a package manager that can be accessed by loading the `Pkg` package:
```julia
julia> using Pkg
```
You generally do not need to load this package, unless you wish to inteface programmatically with the package manager.
The package manager can also be accessed interactively by pressing the `]` key from the Julia REPL prompt, which opens the Pkg REPL:
```julia
julia>]
(@v1.11) pkg> add BenchmarkTools
```
Notice the change of prompt indicator: `v1.11` refers to the currently active environment, which *by default* is the global environment for that current version of Julia (in this case v1.11).
You should generally *not* add packages to this environment, as they will be available to *every* project using Julia v1.11. 
You should instead restrict yourself to installing only tooling packages, such as the [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) package.

For packages that go beyond simple tooling, a specific environment should *always* be used. 
Suppose we are working in the directory `MyProject`, then an environment associated with this project can be activated from Pkg prompt using
```julia
(@v1.11) pkg> activate .
(MyProject) pkg> add Plots
```
where we have then added the [Plots](https://github.com/JuliaPlots/Plots.jl) package to this environment (again notice the prompt change).
In contrast to Python's `venv`, Julia environments are very lightweight and consist of two files:

- `Project.toml`: lists all the packages used by the environment, which can be considered a list of dependencies if the environment represents a package. 
- `Manifest.toml`: is machine-generated file that records the entire dependency tree down to the exact versions of all packages, ensuring the environment can be exactly reproduced.


Note that neither file is created automatically if no packages have been added. Simply executing `julia>] activate .` is not enough to generate them if they do not already exist. An environment can also be activate from the command line when launching Julia:
```bash
julia --project=. myscript.jl
```
which executes `myscript.jl` within the environment associated with the current working directory. This is where loading the package Pkg can be useful. One can for example first run:
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```
to make sure all packages appearing in the Manifest.toml are downloaded and precompiled at the correct versions.

## Development workflow and Revise.jl

Suppose you are editing a file `myadd.jl` containing the code:
```julia
# myadd.jl
function myadd(a, b)
    return a - b
end
```
A sensible workflow is to open a Julia session and include the code contained in this file into the session:
```julia
julia> include("myadd.jl")
```
The function `myadd` can now be tested:
```julia
julia> myadd(2, 3)
-1
```
It appears this function might not be doing what it intented. 
If, without exiting Julia, you make the fix:
```julia
# myadd.jl
function myadd(a, b)
    return a + b # This should now correctly add a and b.
end
```
you will notice this change is not reflected in the REPL:

```julia
julia> myadd(2, 3)
-1
```
Normally, one would need to restart Julia and then `include` the package again, which is impractical when testing code. To address this, there is an essential package not included in the Julia standard library: [Revise.jl](https://timholy.github.io/Revise.jl/stable/). Revise.jl automatically updates any changes to source code loaded in the REPL without requiring a restart, provided the file is loaded using:
```julia
using Revise
julia> Revise.includet("myadd.jl") # notice the `t` in `includet`
```
Changes to locally included modules using `using Module` are also updated without a restart, provided Revise.jl is loaded first.
The Revise.jl package is a prototypical example of a package that *should* be included in your global environment. 
You could even go as far as automatically loading it everytime the Julia is launched by adding
```julia
# ~/.julia/config/startup.jl
using Revise
```
to the your `startup.jl`. The code in this file is executed whenever Julia is started, and can be disabled using:
```bash
julia --startup-file=no
```

# General Purpose GPU Programming

## CUDA



When loading the `CUDA` package, Julia will by default attempt to download a suitable version of the CUDA toolkit based on the devices it detects. On compute nodes, however, internet access is typically unavailable, so this approach fails. Therefore, Julia must be instructed to use a locally installed CUDA toolkit instead of attempting to download binaries.

Assume that [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) v12.2 is installed on the HPC cluster by the system administrator. Launch a Julia REPL and execute:
```julia
julia> using CUDA
julia> CUDA.set_runtime_version!(v"12.2"; local_toolkit=true)
```
This creates a file named `LocalPreferences.toml` in the working directory, instructing Julia to use the locally available CUDA toolkit.

Passing the version to `CUDA.set_runtime_version!` is not strictly necessary, but it ensures that packages can precompile correctly and may be required for some to work properly. The version specified should match the CUDA runtime installed on the node. Your system administrator can provide this information, or you can compile and run a small program to query the available devices and the installed CUDA toolkit version.

The version passed should match the CUDA runtime version installed on the node. The system admin should be able to tell you this information, or you can compile and run this small programme to print information about he devices available and CUDA toolkit version information.

## Other GP-GPU APIs

The Apple Silicon Metal programming framework can be interfaced with [Metal.jl](https://github.com/JuliaGPU/Metal.jl), ROCm (for AMD GPUs) with [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), and Intel's oneAPI with [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl). 
There also exists [KernalAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl?tab=readme-ov-file) which allows one to write GPU kernals *agnostic* to the backend, which may be of interest.

# MPI

The Message Passing Inferface (MPI) interface can be accessed using the [MPI.jl](https://github.com/JuliaParallel/MPI.jl) package. 
You should follow the [configuration](https://juliaparallel.org/MPI.jl/stable/configuration/) guide as, similar to accessing the CUDA interface, Julia needs to know which implementation of MPI are available on the system.
Similarly to the CUDA case, MPI.jl *will* attempt to download and install an implementation, however this will fail if Julia is executing on a HPC compute node without internet access.

# Other Useful Packages

## JLD2

[JLD2](https://github.com/JuliaIO/JLD2.jl) is a very useful data format for saving the outputs of simulations and arbitrary custom Julia data structures. *It is however not recommend for long-term data storage*.
JLD2 relies on the stored data structures being available in the namespace, so make sure you load the required packages containing those data structures before loading the `.jld2` files. 
If any stored data structures have now changed (by a package update, for example) since writing he file, then this will cause issues that need to be patched. See [this part](https://juliaio.github.io/JLD2.jl/dev/advanced/) documentation.

To avoid any issues when saving using JLD2, pin the relevant packages and *always* commit a `Manifest.toml` so that the exact state of the data structures can be recovered, and do not write functions to JLD2 files.
For long term storage and interoperability, convert data structures to be compatible with [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) using.


## DrWatson

[DrWatson](https://github.com/JuliaDynamics/DrWatson.jl) can be useful for managing scientific projects.
While the [project template](https://juliadynamics.github.io/DrWatson.jl/dev/project/#Project-Setup) is often overkill, the package exports some useful functions that can be used standalone.

When loaded alongside the DataFrames package, the [`collect_results!`](https://juliadynamics.github.io/DrWatson.jl/dev/run&list/#DrWatson.collect_results!) function becomes available which can be used to recurse through a directory appending the data, in the form of key-value pairs, contained in `.jld2` files to a data frame which gets subsequently gets written to a file.
When running time-evolution simulations, a useful pipeline is to save a file for each time step in a directory.
For example, suppose we have are running a simulation of the dynamics of a system that we expect to reach a steady state. 
After 100 time steps for two different simulations, we may have a directory structure that looks like:
```
- data
    - raw
        - sim1
            - step_1.jld2
            - step_2.jld2
            - ...
            - step_100.jld2
        - sim2
            - step_1.jld2
            - step_2.jld2
            - ...
            - step_100.jld2
```
One can then call `collect_results!` in each of these directories to save a data frame in the same directory to the file `dynamics.jl`
```julia
function search_dynamics!(list_of_dirs)
    for dir in list_of_dirs
        # Write data frame somewhere in the "processed data" directory.
        dynamics_path = datadir("pro", dir, "dynamics.jld2")

        collect_results!(dynamics_path, dir)
    end
end
search_dynamics!(["sim1", "sim2"])
```
Then one has:
```
- data
    - pro 
        - sim1
            - dynamics.jld2
        - sim2
            - dynamics.jld2
    - raw
        - ...
```
in addition to the contents of the `raw` directory.
Suppose each of the results files in the `raw` directory has a key `time` that stores the current time-axis variable for that simulation time step.
Then one can collect the *final row* of each of the `dynamics.jld2` data frames into a new data frame storing the steady state of *each* simulation.
```julia
function steady_state!()
    # Construct an anonymous function for getting the final row of a data frame
    last_row_only = data -> begin

        # `collect_results!` writes the data frame with key `df` in the .jld2 file.
        df = data["df"]

        sort!(df, "time")

        final_time_step = df[end, :]

        # `collect_results!` expects a vector of pairs
        return collect(pairs(final_time_step))
    end

    steady_state_path = datadir("pro", "steady-state.jld2"),

    # The directory to search, this is where we put our `dynamics.jld2` files from earlier
    dir = datadir("pro");

    df = collect_results!(
        steady_state_path,
        dir;
        subfolders = true,              # Search through all the subfolders of `dir`
        rinclude = [r"dynamics.jld2"],  # Only look in files matching this name
        white_list = [],                # Include none of the data by default
        special_list = [last_row_only], # Function to call on the output of `load`
    )

    return df
end
```
After executing the above function, we now we have an additional file in the `pro` data directory storing *just* the steady state:
```
- data
    - pro 
        - sim1
            - dynamics.jld2
        - sim2
            - dynamics.jld2
        - steady-state.jld2
    - raw
        - ...
```
We can still access the dynamics as this file name is automatically saved in the `path` key of the data frame in `steady-state.jld2`:
```julia
df = steady_state!()
# Get the dynamics of the simulation stored in the first row
dynamics = load(df[1, "path"])["df"]
```
This avoids large data frames (containing *all* the data for *each* time step) which can be difficult to load into RAM.

This is such an example of a workflow possible with DrWatson.

# Adding private Github packages using Pkg
If you wish to add a private repo to your environment using Pkg, then the easiest way to authenticate is via the [Github CLI](https://cli.github.com). 
First install this and follow through with the configuration, and then make sure the following environment variable is set whenever you wish to `add` a private Github repo:
```bash
export JULIA_PKG_USE_CLI_GIT=true
```
You should put this in you `.bashrc` (or whatever the corresponding configuration file is for your shell of choice)

# Further Reading
- [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/) from the official Julia Documentation. This is essential reading to avoid common pitfalls related to performance.
- [Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/) provides context to certain conventions and common themes among Julia code. You should stick to it as much as possiblebl.
- [Modern Julia Workflows](https://modernjuliaworkflows.org) contains similar guides to those hosted here.
