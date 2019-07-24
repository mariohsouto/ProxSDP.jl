path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
using ProxSDP, JuMP

# Create a JuMP model using ProxSDP as the solver
model = Model(with_optimizer(ProxSDP.Optimizer, log_verbose=true, tol_primal = 1e-5, tol_dual = 1e-5))

@variable(model, X[1:2,1:2], PSD)
@variable(model, Y[1:3,1:3], PSD)

x = X[1,1]
y = X[2,2]

@constraint(model, ub_x, x <= 2)
@constraint(model, ub_y, y <= 30)
@constraint(model, con, 1x + 5y <= 3)

@constraint(model, test, Y[1, 1] + 5y <= 3)

# ProxSDP supports maximization or minimization
# of linear functions
@objective(model, Max, 5x + 3 * y)

# Then we can solve the model
JuMP.optimize!(model)

# And ask for results!
JuMP.objective_value(model)

JuMP.value(x)

JuMP.value(y)