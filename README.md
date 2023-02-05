![ProxSDP logo](https://github.com/mariohsouto/ProxSDP_aux/blob/master/logo_proxSDP.png "ProxSDP logo")
---

| **Build Status** |
|:-----------------:|
| [![Build Status][build-img]][build-url] [![Codecov branch][codecov-img]][codecov-url] [![][docs-img]][docs-url]|

[build-img]: https://github.com/mariohsouto/ProxSDP.jl/workflows/CI/badge.svg?branch=master
[build-url]: https://github.com/mariohsouto/ProxSDP.jl/actions?query=workflow%3ACI
[codecov-img]: http://codecov.io/github/mariohsouto/ProxSDP.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/mariohsouto/ProxSDP.jl?branch=master
[docs-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-url]: https://mariohsouto.github.io/ProxSDP.jl/latest/

**ProxSDP** is an open-source semidefinite programming ([SDP](https://en.wikipedia.org/wiki/Semidefinite_programming)) solver based on the paper ["Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting"](https://arxiv.org/abs/1810.05231). The main advantage of ProxSDP over other state-of-the-art solvers is the ability to exploit the **low-rank** structure inherent to several SDP problems.

### Overview of problems ProxSDP can solve

* General conic convex optimization problems with the presence of the [positive semidefinite cone](https://web.stanford.edu/~boyd/papers/pdf/semidef_prog.pdf), [second-order cone](https://web.stanford.edu/~boyd/papers/pdf/socp.pdf) and [positive orthant](https://www.math.ucla.edu/~tom/LP.pdf);
* Semidefinite relaxation of nonconvex problems, e.g. [max-cut](http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf), [binary MIMO](https://arxiv.org/pdf/cs/0606083.pdf), [optimal power flow](http://authorstest.library.caltech.edu/141/1/TPS_OPF_2_tech.pdf), [sensor localization](https://web.stanford.edu/~boyd/papers/pdf/sensor_selection.pdf), [sum-of-squares](https://en.wikipedia.org/wiki/Sum-of-squares_optimization);
* Control theory problems with [LMI](https://en.wikipedia.org/wiki/Linear_matrix_inequality) constraints;
* Nuclear norm minimization problems, e.g. [matrix completion](https://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf);

## Installation

You can install **ProxSDP** through the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html):
```julia
] add ProxSDP
```

## Usage

Let's consider the semidefinite programming relaxation of the [max-cut](http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf) problem as in
```
    max   0.25 * W•X
    s.t.  diag(X) = 1,
          X ≽ 0,
```

### JuMP
This problem can be solved by the following code using **ProxSDP** and [JuMP](https://github.com/JuliaOpt/JuMP.jl).
```julia
# Load packages
using ProxSDP, JuMP, LinearAlgebra

# Number of vertices
n = 4
# Graph weights
W = [18.0  -5.0  -7.0  -6.0
     -5.0   6.0   0.0  -1.0
     -7.0   0.0   8.0  -1.0
     -6.0  -1.0  -1.0   8.0]

# Build Max-Cut SDP relaxation via JuMP
model = Model(ProxSDP.Optimizer)
set_optimizer_attribute(model, "log_verbose", true)
set_optimizer_attribute(model, "tol_gap", 1e-4)
set_optimizer_attribute(model, "tol_feasibility", 1e-4)

@variable(model, X[1:n, 1:n], PSD)
@objective(model, Max, 0.25 * dot(W, X))
@constraint(model, diag(X) .== 1)

# Solve optimization problem with ProxSDP
optimize!(model)

# Retrieve solution
Xsol = value.(X)
```
### Convex.jl
Another alternative is to use **ProxSDP** via [Convex.jl](https://github.com/jump-dev/Convex.jl) as the following
```julia
# Load packages
using Convex, ProxSDP

# Number of vertices
n = 4
# Graph weights
W = [18.0  -5.0  -7.0  -6.0
     -5.0   6.0   0.0  -1.0
     -7.0   0.0   8.0  -1.0
     -6.0  -1.0  -1.0   8.0]
     
# Define optimization problem
X = Semidefinite(n)
problem = maximize(0.25 * dot(W, X), diag(X) == 1)

# Solve optimization problem with ProxSDP
solve!(problem, ProxSDP.Optimizer(log_verbose=true, tol_gap=1e-4, tol_feasibility=1e-4))

# Get the objective value
problem.optval

# Retrieve solution
evaluate(X)
```

## Citing this package

The published version of the paper can be found [here](https://doi.org/10.1080/02331934.2020.1823387) and the arXiv version [here](https://arxiv.org/pdf/1810.05231.pdf).

We kindly request that you cite the paper as:

```bibtex
@article{souto2020exploiting,
  author = {Mario Souto and Joaquim D. Garcia and \'Alvaro Veiga},
  title = {Exploiting low-rank structure in semidefinite programming by approximate operator splitting},
  journal = {Optimization},
  pages = {1-28},
  year  = {2020},
  publisher = {Taylor & Francis},
  doi = {10.1080/02331934.2020.1823387},
  URL = {https://doi.org/10.1080/02331934.2020.1823387}
}
```

The preprint version of the paper can be found [here](https://arxiv.org/abs/1810.05231).

## Disclaimer

* ProxSDP is a research software, therefore it should not be used in production.
* Please open an issue if you find any problems, developers will try to fix and find alternatives.
* There is no continuous development for 32-bit systems, the package should work, but might reach some issues.
* ProxSDP assumes primal and dual feasibility.

## ROAD MAP

- Support for exponential and power cones;
- Warm start.
