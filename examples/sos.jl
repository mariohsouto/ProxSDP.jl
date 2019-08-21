path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
using SumOfSquares
using DynamicPolynomials
using ProxSDP, SCS

# Using ProxSDP as the SDP solver
model = SOSModel(with_optimizer(ProxSDP.Optimizer, log_verbose=true, max_iter=100000))
# model = SOSModel(with_optimizer(SCS.Optimizer, max_iters=100000))

@polyvar x z
@variable(model, t)
p = x^4 + x^2 - 3*x^2*z^2 + z^6 - t
@constraint(model, p >= 0)
@objective(model, Max, t)
optimize!(model)
println("Solution: $(value(t))")
# Returns the lower bound -.17700

