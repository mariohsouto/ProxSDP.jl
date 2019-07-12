path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__), "data")

is_julia1 = VERSION >= v"1.0"
if is_julia1
    using Test
    using Dates
    using Random
    using LinearAlgebra
    using DelimitedFiles
    using SparseArrays
else
    using Base.Test
end

use_MOI = false
sets_to_test = Symbol[]
# push!(sets_to_test, :RANDSDP)
# push!(sets_to_test, :SENSORLOC)
# push!(sets_to_test, :SDPLIB)
push!(sets_to_test, :MIMO)


@static if use_MOI
    using ProxSDP, MathOptInterface
    using MosekTools
    if is_julia1
        LinearAlgebra.symmetric_type(::Type{MathOptInterface.VariableIndex}) = MathOptInterface.VariableIndex
        LinearAlgebra.symmetric(v::MathOptInterface.VariableIndex, ::Symbol) = v
        LinearAlgebra.transpose(v::MathOptInterface.VariableIndex) = v
    end
    include("moi_init.jl")
    # optimizer = MOIU.CachingOptimizer(ProxSDPModelData{Float64}(), ProxSDP.Optimizer(log_verbose=true, timer_verbose = true))
    optimizer = ProxSDP.Solver(log_verbose=true, timer_verbose = true, convergence_window=200) #, tol_primal = 1e-3, tol_dual = 1e-3)
else
    solvers = Tuple{String, Function}[]
    using JuMP
    using ProxSDP
    push!(solvers, ("ProxSDP", () -> ProxSDP.Optimizer(log_verbose=true, timer_verbose = true, time_limit = 900.0, log_freq = 1_000)))
    # push!(solvers, ("ProxSDP_fullrank", () -> ProxSDP.Optimizer(full_eig_decomp = true, log_verbose=true, timer_verbose = false, time_limit = 900.0, log_freq = 1_000)))
    # using MosekTools
    # push!(solvers, ("MOSEK", () -> Mosek.Optimizer(MSK_DPAR_OPTIMIZER_MAX_TIME = 900.0))) # eps = ???
    # using CSDP
    # push!(solvers, ("CSDP", () -> CSDP.Optimizer(objtol=1e-4, maxiter=100000))) # eps = ???
    # using SDPA
    # push!(solvers, ("SDPA", () -> SDPA.Optimizer())) # eps = ???
    # using SCS
    # push!(solvers, ("SCS", () -> SCS.Optimizer())) # eps = 1e-4
    # using COSMO
    # push!(solvers, ("COSMO", () -> COSMO.Optimizer(time_limit = 900.0, max_iter = 100_000, eps_abs = 1e-3))) # eps = 1e-4
end

# function ProxSDP.get_solution(opt)
#     nothing
# end

NOW = is_julia1 ? replace("$(now())",":"=>"_") : replace("$(now())",":","_")
FILE = open(joinpath(dirname(@__FILE__),"proxsdp_bench_$(NOW).log"),"w")
println(FILE, "class, prob_ref, time, p_obj, d_obj, p_res, d_res")
function println2(FILE, class::String, ref::String, sol::ProxSDP.MOISolution)
    println(FILE, "$class, $ref, $(sol.time), $(sol.objval), $(sol.dual_objval), $(sol.primal_residual), $(sol.dual_residual)")
    flush(FILE)
end
println2(FILE, class::String, ref::String, sol::Nothing) = nothing
function println2(FILE, class::String, ref::String, sol)
    println(FILE, "$class, $ref, $(sol[2]), $(sol[1])")
    flush(FILE)
end
function println2(FILE, solver::String, class::String, ref::String, sol)
    # println(FILE, "$solver, $class, $ref, $(sol[2]), $(sol[1])")
    println(FILE, "$solver, $class, $ref, $(sol[2]), $(sol[1]), $(sol[3])")
    flush(FILE)
end

RANDSDP_TEST_SET = 1:1
SENSORLOC_TEST_SET = [#100:100:300#1000
    100,
    200,
    300,
    # 400, # mosek stops here - maybenot in new form
    # 500, # proxsdp 166s
    # 600,
    # 700,
    # 800,
    # 900,
    # 1000,
]
MIMO_TEST_SET = [
    100,
    # 500, # mosek stops here
    1000, # 700s on scs
    # 1500, # SCS stops here (?)
    # 2000,
    # 2500,
    # 3000,
    # 3500,
    # 4000,
    # 4500,
    # 5000,
    # 6000,
    # 7000,
    # 8000,
    # 9000,
    # 10_000,
    ]
GPP_TEST_SET = [
    "gpp124-1.dat-s",
    "gpp124-1.dat-s",
    "gpp124-2.dat-s",
    "gpp124-3.dat-s",
    "gpp124-4.dat-s",
    "gpp250-1.dat-s",
    "gpp250-2.dat-s",
    "gpp250-3.dat-s",
    "gpp250-4.dat-s",
    "gpp500-1.dat-s", # SCS > 1000 s
    "gpp500-2.dat-s", # SCS > 1000 s
    "gpp500-3.dat-s",
    "gpp500-4.dat-s",
    # "equalG11.dat-s",
    # "equalG51.dat-s",
]
MAXCUT_TEST_SET = [
    "mcp124-1.dat-s",
    "mcp124-1.dat-s",
    "mcp124-2.dat-s",
    "mcp124-3.dat-s",
    "mcp124-4.dat-s",
    "mcp250-1.dat-s",
    "mcp250-2.dat-s",
    "mcp250-3.dat-s",
    "mcp250-4.dat-s",
    "mcp500-1.dat-s",
    "mcp500-2.dat-s",
    "mcp500-3.dat-s",
    "mcp500-4.dat-s",
    # "maxG11.dat-s"  ,
    # "maxG51.dat-s"  ,
    # "maxG32.dat-s"  ,
    # "maxG55.dat-s"  ,
    # "maxG60.dat-s"  ,
]

include("base_randsdp.jl")
include("moi_randsdp.jl")
include("jump_randsdp.jl")

include("base_mimo.jl")
include("moi_mimo.jl")
include("jump_mimo.jl")

include("base_sensorloc.jl")
include("moi_sensorloc.jl")
include("jump_sensorloc.jl")

include("base_sdplib.jl")
include("moi_sdplib.jl")
include("jump_sdplib.jl")

if use_MOI
    _randsdp = moi_randsdp
    _mimo = moi_mimo
    _sensorloc = moi_sensorloc
    _sdplib = moi_sdplib
else
    _randsdp = jump_randsdp
    _mimo = jump_mimo
    _sensorloc = jump_sensorloc
    _sdplib = jump_sdplib
end

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


close(FILE)