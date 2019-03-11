# Manual

## Building problems with JuMP.jl

Currently the easiest ways to pass problems to ProxSDP is through [JuMP](https://github.com/JuliaOpt/JuMP.jl) or MathOptInterface (v0.8).

The main caveat is that currently ProxSDP must have one and only one PSD variable, no other variables are allowed.

In the test folder one can find MOI implementations of some problems: MIMO, Sensor Localization, Random SDPs and sdplib problems.

## Solver arguments

Argument | Description | Type | Values (default)
--- | --- | --- |  ---
log_verbose | print evolution of the process | `Bool` |  `false`
log_freq | print evoluition of the process every n iterations | `Int` |  `100`
timer_verbose | #TODO | `Bool` |  `false`
max_iter | #TODO | `Int` |  `100000`
tol_primal | #TODO | `Float64` |  `1e-3`
tol_dual | #TODO | `Float64` |  `1e-3`
tol_eig | #TODO | `Float64` |  `1e-3`
tol_soc | #TODO | `Float64` |  `1e-3`
initial_theta | #TODO | `Float64` |  `1.0`
initial_beta | #TODO | `Float64` |  `1.0`
min_beta | #TODO | `Float64` |  `1e-3`
max_beta | #TODO | `Float64` |  `1e+3`
initial_adapt_level | #TODO | `Float64` |  `0.9`
adapt_decay | #TODO | `Float64` |  `0.9`
convergence_window | #TODO | `Int` |  `100`
convergence_check | #TODO | `Int` |  `50`
residual_relative_diff | #TODO | `Float64` |  `100.0`
max_linsearch_steps | #TODO | `Int` |  `1000`
full_eig_decomp | #TODO | `Bool` |  `false`
max_target_rank_krylov_eigs | #TODO | `Int` |  `100`
min_size_krylov_eigs | #TODO | `Int` |  `16`

## JuMP example

A quick JuMP example:

```julia
using ProxSDP, JuMP

# Create a JuMP model using ProxSDP as the solver
model = Model(with_optimizer(ProxSDP.Optimizer, log_verbose=true))

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

JuMP.value(x)

JuMP.value(y)
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
