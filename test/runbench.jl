#=
    Set path data
=#

path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__), "data")

#=
    Load required libraries
=#

using Test
using Dates
using Random
using DelimitedFiles
using SparseArrays
using JuMP

#=
    select problem types to be tested
=#
sets_to_test = Symbol[]
push!(sets_to_test, :RANDSDP)
push!(sets_to_test, :SENSORLOC)
push!(sets_to_test, :SDPLIB)
push!(sets_to_test, :MIMO)

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
    time_limit = 10.0,#900.0,
    log_freq = 1_000,
    )))

#=
    ProxSDP with defautl parameters excpet for FULL RANK decomposition
=#
# using ProxSDP
# push!(solvers, ("ProxSDPfull", () -> ProxSDP.Optimizer(
#     log_verbose=true,
#     time_limit = 900.0,
#     log_freq = 1_000,
#     full_eig_decomp = true,
#     )))

#=
    First order solvers
=#

# using SCS
# push!(solvers, ("SCS", () -> SCS.Optimizer(eps = 1e-4))) # eps = 1e-4

# using COSMO
# push!(solvers, ("COSMO", () -> COSMO.Optimizer(time_limit = 900.0, max_iter = 100_000, eps_abs = 1e-4))) # eps = 1e-4

# add SDPNAL+
# using SDPNAL
# push!(solvers, ("SDPNAL", () -> SDPNAL.Optimizer())) # eps = 1e-4

#=
    Interior point solvers
=#

# using MosekTools
# push!(solvers, ("MOSEK", () -> Mosek.Optimizer(MSK_DPAR_OPTIMIZER_MAX_TIME = 900.0)))

# using CSDP
# push!(solvers, ("CSDP", () -> CSDP.Optimizer(objtol=1e-4, maxiter=100000)))

# using SDPA
# push!(solvers, ("SDPA", () -> SDPA.Optimizer()))

#=
    Funstions to write results
=#

NOW = replace("$(now())",":"=>"_")
FILE = open(joinpath(dirname(@__FILE__),"proxsdp_bench_$(NOW).log"),"w")

println(FILE, "class, prob_ref, time, obj, rank, lin_feas, sdp_feas")

function println2(FILE, solver::String, class::String, ref::String, sol)
    println(FILE, "$solver, $class, $ref, $(sol[2]), $(sol[1]), $(sol[3]), $(sol[4]), $(sol[5]), $(sol[6])")
    flush(FILE)
end

#=
    Selection of problem instances
=#

RANDSDP_TEST_SET = 1:1
SENSORLOC_TEST_SET = [
    100,
    200,
    300,
    400,
]
MIMO_TEST_SET = [
    100,
    500,
    1000,
    1500,
    2000,
    ]
GPP_TEST_SET = [
    "gpp124-1.dat-s",
    "gpp124-1.dat-s",
    "gpp124-2.dat-s",
    "gpp124-3.dat-s",
    "gpp124-4.dat-s",

    # "gpp250-1.dat-s",
    # "gpp250-2.dat-s",
    # "gpp250-3.dat-s",
    # "gpp250-4.dat-s",
    # "gpp500-1.dat-s",
    # "gpp500-2.dat-s",
    # "gpp500-3.dat-s",
    # "gpp500-4.dat-s",

    # "equalG11.dat-s",
    # "equalG51.dat-s",
]
MAXCUT_TEST_SET = [
    "mcp124-1.dat-s",
    "mcp124-1.dat-s",
    "mcp124-2.dat-s",
    "mcp124-3.dat-s",
    "mcp124-4.dat-s",

    # "mcp250-1.dat-s",
    # "mcp250-2.dat-s",
    # "mcp250-3.dat-s",
    # "mcp250-4.dat-s",
    # "mcp500-1.dat-s",
    # "mcp500-2.dat-s",
    # "mcp500-3.dat-s",
    # "mcp500-4.dat-s",

    # "maxG11.dat-s"  ,
    # "maxG51.dat-s"  ,
    # "maxG32.dat-s"  ,
    # "maxG55.dat-s"  ,
    # "maxG60.dat-s"  ,
]

#=
    Load problem testing functions
=#

include("base_randsdp.jl")
include("jump_randsdp.jl")

include("base_mimo.jl")
include("jump_mimo.jl")

include("base_sensorloc.jl")
include("jump_sensorloc.jl")

include("base_sdplib.jl")
include("jump_sdplib.jl")

_randsdp = jump_randsdp
_mimo = jump_mimo
_sensorloc = jump_sensorloc
_sdplib = jump_sdplib

#=
    Run benchmarks
=#

for optimizer in solvers
    if :RANDSDP in sets_to_test
        println("RANDSDP")
        _randsdp(optimizer[2], 0, 5, 5)
        for i in RANDSDP_TEST_SET
            @show i
            sol = _randsdp(optimizer[2], i, 5, 5)
            println2(FILE, optimizer[1], "RANDSDP", "$i", sol)
        end
    end
    if :SENSORLOC in sets_to_test
        println("SENSORLOC")
        for n in SENSORLOC_TEST_SET
            @show n
            sol = _sensorloc(optimizer[2], 0, n)
            println2(FILE, optimizer[1], "SENSORLOC", "$n", sol)
        end
    end
    if :SDPLIB in sets_to_test
        println("gpp")
        for name in GPP_TEST_SET
            println(name)
            sol = _sdplib(optimizer[2], joinpath(datapath, name))
            println2(FILE, optimizer[1], "SDPLIB_gp", name, sol)
        end
        println("max_cut")
        for name in MAXCUT_TEST_SET
            println(name)
            sol = _sdplib(optimizer[2], joinpath(datapath, name))
            println2(FILE, optimizer[1], "SDPLIB_mc", name, sol)
        end
    end
    if :MIMO in sets_to_test
        println("MIMO")
        _mimo(optimizer[2], 0, 100)
        for n in MIMO_TEST_SET
            @show n
            sol = _mimo(optimizer[2], 0, n)
            println2(FILE, optimizer[1], "MIMO", "$n", sol)
        end
    end
end

#=
    Finish benhcmark file
=#

close(FILE)