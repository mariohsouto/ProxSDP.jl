
# only Convex.jl tests
# using Convex, ProxSDP, Test
# using Convex.ProblemDepot: run_tests
# @testset "Convex Problem Depot tests" begin
#     run_tests(;  exclude=[r"mip", r"exp"]) do problem
#         solve!(problem, () -> ProxSDP.Optimizer(
#             log_freq = 1_000_000, log_verbose = true,
#             tol_gap = 5e-8, tol_feasibility = 1e-7,
#             max_iter = 10_000_000, time_limit = 30.)
#             )
#     end
# end
path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__), "data")

using ProxSDP
# Convex.jl and SumOfSquares.jl tests
using ConvexTests, ProxSDP
@info "Starting ProxSDP tests"
do_tests("ProxSDP", () -> ProxSDP.Optimizer(
    log_freq = 1_000_000, log_verbose = true,
    tol_gap = 5e-8, tol_feasibility = 1e-7,
    max_iter = 100_000_000, time_limit = 4 * 30.
); exclude = [r"mip", r"exp"])
