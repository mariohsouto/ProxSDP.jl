# ProxSDP

| **Build Status** |
|:-----------------:|
| [![Build Status][build-img]][build-url] [![Codecov branch][codecov-img]][codecov-url] |

[build-img]: https://travis-ci.org/mariohsouto/ProxSDP.jl.svg?branch=master
[build-url]: https://travis-ci.org/mariohsouto/ProxSDP.jl
[codecov-img]: http://codecov.io/github/mariohsouto/ProxSDP.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/mariohsouto/ProxSDP.jl?branch=master

ProxSDP is a semidefinite programming ([SDP](https://en.wikipedia.org/wiki/Semidefinite_programming)) solver based on the paper "Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting". ProxSDP solves general SDP problems by means of a first order proximal algorithm based on the [primal-dual hybrid gradient](http://www.cmapx.polytechnique.fr/preprint/repository/685.pdf), also known as Chambolle-Pock method. The main advantage of ProxSDP over other state-of-the-art solvers is the ability of exploit the low-rank property inherent to several SDP problems.

### Overview of problems ProxSDP can solve

* Any semidefinite programming problem in [standard form](http://web.stanford.edu/~boyd/papers/pdf/semidef_prog.pdf);
* Semidefinite relaxations of nonconvex problems, e.g. [max-cut](http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf), [binary MIMO](https://arxiv.org/pdf/cs/0606083.pdf), [optimal power flow](http://authorstest.library.caltech.edu/141/1/TPS_OPF_2_tech.pdf), [sensor localization](https://web.stanford.edu/~boyd/papers/pdf/sensor_selection.pdf);
* Nuclear norm minimization problems, e.g. [matrix completion](https://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf).

### Installing

Currently ProxSDP only works with **Julia 0.6.x**

To add ProxSDP run:

`Pkg.add("ProxSDP")`

### Building problems with JuMP.jl

Currently the easiest ways to pass problems to ProxSDP is through [JuMP](https://github.com/JuliaOpt/JuMP.jl) ([v0.19-alpha](https://discourse.julialang.org/t/first-alpha-release-of-jump-0-19-jump-mathoptinterface/16099)) or MathOptInterface (v0.6).

The main caveat is that currently ProxSDP must have one and only one PSD variable, no other variables are allowed.

In the test folder one can find MOI implementations of some problems: MIMO, Sensor Localization, Random SDPs and sdplib problems.

#### JuMP example

ProxSDP uses the new implementation of JuMP, currently in alpha version.

Therefore one needs to checkout in JuMP on the tag [v0.19-alpha](https://discourse.julialang.org/t/first-alpha-release-of-jump-0-19-jump-mathoptinterface/16099)

For more example on how to use the latest version of JuMP please refer to the [manual](http://www.juliaopt.org/JuMP.jl/latest/).

A quick JuMP example:

```julia
using ProxSDP, JuMP

# Create a JuMP model using ProxSDP as the solver
model = Model(with_optimizer(ProxSDP.Solver, log_verbose=true))

# Create a Positive Semidefinite variable
# Currently ProxSDP is only able to hold one PSD
# variable and no other variable
@variable(model, X[1:2,1:2], PSD)

# but you can define pieces of the one PSD
# variable to hold other variable as in the
# Canonical SDP format
x = X[1,1]
y = X[2,2]

# There is no limits on linear constraints
# one can define as many as wanted
@constraint(model, ub_x, x <= 2)

@constraint(model, ub_y, y <= 30)

@constraint(model, con, 1x + 5y <= 3)

# ProxSDP supports maximization or minimization
# of linear functions
@objective(model, Max, 5x + 3 * y)

# Then we can solve the model
JuMP.optimize!(model)

# And ask for results!
JuMP.objective_value(model)

JuMP.result_value(x)

JuMP.result_value(y)
```

### Referencing

The first version of the paper can be found [here](https://arxiv.org/abs/1810.05231).

```
@article{souto2018exploiting,
  title={Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting},
  author={Souto, Mario and Garcia, Joaquim D and Veiga, {\'A}lvaro},
  journal={arXiv preprint arXiv:1810.05231},
  year={2018}
}
```

### TODO

- Add support for multiple SDP Variables
- Add support for scalar Variables
- Add support for other cones
