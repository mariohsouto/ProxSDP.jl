# ProxSDP

| **Build Status** |
|:-----------------:|
| [![Build Status][build-img]][build-url] [![Codecov branch][codecov-img]][codecov-url] [![][docs-img]][docs-url]|

[build-img]: https://travis-ci.org/mariohsouto/ProxSDP.jl.svg?branch=master
[build-url]: https://travis-ci.org/mariohsouto/ProxSDP.jl
[codecov-img]: http://codecov.io/github/mariohsouto/ProxSDP.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/mariohsouto/ProxSDP.jl?branch=master
[docs-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-url]: https://mariohsouto.github.io/ProxSDP.jl/latest/

**ProxSDP** is an open-source semidefinite programming ([SDP](https://en.wikipedia.org/wiki/Semidefinite_programming)) solver based on the paper ["Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting"](https://arxiv.org/abs/1810.05231). The main advantage of ProxSDP over other state-of-the-art solvers is the ability to exploit the **low-rank** structure inherent to several SDP problems.

### Overview of problems ProxSDP can solve

* Semidefinite relaxations of nonconvex problems, e.g. [max-cut](http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf), [binary MIMO](https://arxiv.org/pdf/cs/0606083.pdf), [optimal power flow](http://authorstest.library.caltech.edu/141/1/TPS_OPF_2_tech.pdf), [sensor localization](https://web.stanford.edu/~boyd/papers/pdf/sensor_selection.pdf);
* Nuclear norm minimization problems, e.g. [matrix completion](https://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf);

## Installation

You can install **ProxSDP** through the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html):
```julia
] add ProxSDP
```

## Using ProxSDP with JuMP

For example, the semidefinite programming relaxation of the [max-cut](http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf) problem
```
    max   0.25 * W•X
    s.t.  diag(X) == 1,
          X ≽ 0,
```
can be solved by the following code using **ProxSDP** and [JuMP](https://github.com/JuliaOpt/JuMP.jl).

```julia
# Load packages
using ProxSDP, JuMP

# Build Max-Cut SDP relaxation via JuMP
model = Model(with_optimizer(ProxSDP.Optimizer))
@variable(model, X[1:n, 1:n], PSD)
@objective(model, Max, 0.25 * dot(W, X))
@constraint(model, diag(X) .== 1)

# Solve optimization problem with ProxSDP
JuMP.optimize!(model)

# Retrieve solution
Xsol = JuMP.value.(X)
```

### Referencing

The preprint version of the paper can be found [here](https://arxiv.org/abs/1810.05231).

```
@article{souto2018exploiting,
  title={Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting},
  author={Souto, Mario and Garcia, Joaquim D and Veiga, {\'A}lvaro},
  journal={arXiv preprint arXiv:1810.05231},
  year={2018}
}
```

### ROAD MAP

- Support for exponential and power cones;
- Infeasibility certificate;
- Warm start.
