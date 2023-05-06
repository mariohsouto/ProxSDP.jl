
#=
    Load required libraries
=#

using Random
using JuMP
using LinearAlgebra

#=
    select problem types to be tested
=#
sets_to_test = Symbol[]
push!(sets_to_test, :RANDSDP)
push!(sets_to_test, :SENSORLOC)

#=
    select solvers to be tested
=#
solvers = Tuple{String, Function}[]

#=
    ProxSDP with default parameters
=#
using ProxSDP
push!(solvers, ("ProxSDP", () -> ProxSDP.Optimizer(
    log_verbose=true,
    timer_verbose=true,
    time_limit = 5 * 60.0,
    log_freq = 1_000,
    )))

#=
    Selection of problem instances
=#

RANDSDP_TEST_SET = 1:1
SENSORLOC_TEST_SET = [
    50,
]

#=
    Load problem testing functions
=#

include("base_randsdp.jl")
include("jump_randsdp.jl")
include("base_sensorloc.jl")
include("jump_sensorloc.jl")

_randsdp = jump_randsdp
_sensorloc = jump_sensorloc

#=
    Run benchmarks
=#

for optimizer in solvers
    if :RANDSDP in sets_to_test
        _randsdp(optimizer[2], 0, 10, 10)
        for i in RANDSDP_TEST_SET
            sol = _randsdp(optimizer[2], i, 10, 10)
        end
    end
    if :SENSORLOC in sets_to_test
        for n in SENSORLOC_TEST_SET
            sol = _sensorloc(optimizer[2], 0, n)
        end
    end
end

