path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
using SumOfSquares
using DynamicPolynomials
using ProxSDP, SCS

# Using ProxSDP as the SDP solver
# model = SOSModel(with_optimizer(ProxSDP.Optimizer, log_verbose=true, max_iter=100000, full_eig_decomp=true))
model = SOSModel(ProxSDP.Optimizer)
set_optimizer_attribute(model, "log_verbose", true)
set_optimizer_attribute(model, "max_iter", 100000)
set_optimizer_attribute(model, "full_eig_decomp", true)

@polyvar x z
@variable(model, t)
p = x^4 + x^2 - 3*x^2*z^2 + z^6 - t
@constraint(model, p >= 0)
@objective(model, Max, t)
optimize!(model)
println("Solution: $(value(t))")
# Returns the lower bound -.17700

