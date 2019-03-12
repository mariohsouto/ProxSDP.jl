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