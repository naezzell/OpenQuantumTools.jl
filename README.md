<img src="docs/src/assets/logo.jpg" width="256"/>

# OpenQuantumTools.jl
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://uscqserver.github.io/OpenQuantumTools.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://uscqserver.github.io/OpenQuantumTools.jl/dev/)
[![codecov](https://codecov.io/gh/USCqserver/OpenQuantumTools.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/USCqserver/OpenQuantumTools.jl)

The official name of this package is "Hamiltonian Open Quantum System Toolkit" (HOQST). To conform with the Julia package name [guidelines](https://julialang.github.io/Pkg.jl/v1/creating-packages/) the code name of the package is `OpenQuantumTools`. It is a Julia toolkit for simulating the open quantum system dynamics. The package is still under development, but the current functionality and APIs are stable. Future releases before v1.0 will focus on adding more features instead of introducing breaking changes. Detailed documentation can be found [here](https://uscqserver.github.io/OpenQuantumTools.jl/dev/). Any pull requests are welcome.

## Installation

To install, run the following commands inside the Julia REPL:
```julia
using Pkg
Pkg.add("OpenQuantumTools")
```
Alternatively, this can also be done in Julia's [Pkg REPL](https://julialang.github.io/Pkg.jl/v1/getting-started/):
```julia-REPL
(1.5) pkg> add OpenQuantumTools
```
`OpenQuantumTools` requires Julia 1.4 or higher. Installing it on an older version of Julia will result in an unsatisfiable requirements error.

## Useful Packages
It is recommended to install the following external packages:  
### [Plots.jl](https://github.com/JuliaPlots/Plots.jl)
Plots is a visualization interface and toolset for Julia. `OpenQuantumTools.jl` provides several plotting functionality by recipes to `Plots.jl`.
### [DifferentialEquations.jl](http://docs.juliadiffeq.org/latest/)
Even though `OpenQuantumTools.jl` can function without `DifferentialEquations.jl`, it needs to be loaded in order for the master equation solvers to work properly. For [low dependency usage](http://docs.juliadiffeq.org/stable/features/low_dep.html#Low-Dependency-Usage-1), replacing `DifferentialEquations` by [OrdinaryDiffEq.jl](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl) will also work.

## Tutorials
Tutorials and examples can be found in [HOQSTTutorials.jl](https://github.com/USCqserver/HOQSTTutorials.jl).

## Citing

The corresponding paper for `OpenQuantumTools` is

[[1] H. Chen and D. A. Lidar, HOQST: Hamiltonian Open Quantum System Toolkit, ArXiv:2011.14046 [Quant-Ph] (2020)](https://arxiv.org/abs/2011.14046)

This software is developed as part of academic research. If you use `OpenQuantumTools` as part of your research, teaching, or other activities, we would be grateful if you could cite our work.


## Acknowledgment
The authors thank Grace Chen for the HOQST logo design. The authors also thank Ka-Wa Yip for providing his data and MATLAB program on 1/f noise simulation for cross-checking. You can find them in this [repo](https://github.com/USCqserver/1fnoise).