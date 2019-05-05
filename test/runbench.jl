path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__), "data")
# using JuMP
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
# import Base.is_empty

use_MOI = false
sets_to_test = Symbol[]
push!(sets_to_test, :MIMO)
push!(sets_to_test, :RANDSDP)
push!(sets_to_test, :SDPLIB)
push!(sets_to_test, :SENSORLOC)

@static if use_MOI#Base.libblas_name == "libmkl_rt"
    using ProxSDP, MathOptInterface
    # using MosekTools
    if is_julia1
        LinearAlgebra.symmetric_type(::Type{MathOptInterface.VariableIndex}) = MathOptInterface.VariableIndex
        LinearAlgebra.symmetric(v::MathOptInterface.VariableIndex, ::Symbol) = v
        LinearAlgebra.transpose(v::MathOptInterface.VariableIndex) = v
    end
    include("moi_init.jl")
    # optimizer = MOIU.CachingOptimizer(ProxSDPModelData{Float64}(), ProxSDP.Optimizer(log_verbose=true, timer_verbose = true))
    optimizer = ProxSDP.Solver(log_verbose=false, timer_verbose = false)
    # optimizer = Mosek.Optimizer()
    # @show optimizer = MathOptInterface.Bridges.full_bridge_optimizer(Mosek.Optimizer(), Float64)
else
    using JuMP
    using ProxSDP
    optimizer = () -> ProxSDP.Optimizer(log_verbose=false, timer_verbose = false)

    # using CSDP
    # optimizer = CSDPSolver(objtol=1e-4, maxiter=100000)
    # using SCS
    # optimizer = SCSSolver(eps=1e-4, verbose=true)
    # using MosekTools
    # optimizer = Mosek.Optimizer()

end

function ProxSDP.get_solution(opt)
    nothing
end

NOW = is_julia1 ? replace("$(now())",":"=>"_") : replace("$(now())",":","_")
FILE = open(joinpath(dirname(@__FILE__),"proxsdp_bench_$(NOW).log"),"w")
println(FILE, "class, prob_ref, time, p_obj, d_obj, p_res, d_res")
function println2(FILE, class::String, ref::String, sol::ProxSDP.MOISolution)
    println(FILE, "$class, $ref, $(sol.time), $(sol.objval), $(sol.dual_objval), $(sol.primal_residual), $(sol.dual_residual)")
    flush(FILE)
end
println2(FILE, class::String, ref::String, sol::Nothing) = nothing

RANDSDP_TEST_SET = 1:1
SENSORLOC_TEST_SET = 50:50:300
MIMO_TEST_SET = [100, 500, 1000, 2000, 3000, 4000, 5000]
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
    "gpp500-1.dat-s",
    "gpp500-2.dat-s",
    "gpp500-3.dat-s",
    "gpp500-4.dat-s",
    "equalG11.dat-s",
    "equalG51.dat-s",
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
    "maxG11.dat-s"  ,
    "maxG51.dat-s"  ,
    "maxG32.dat-s"  ,
    "maxG55.dat-s"  ,
    "maxG60.dat-s"  ,
]
if true
    RANDSDP_TEST_SET = 1:1
    SENSORLOC_TEST_SET = 50:50:200#300
    MIMO_TEST_SET = [100, 500, 1000, 2000]#, 3000, 4000, 5000]
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
        "gpp500-1.dat-s",
        "gpp500-2.dat-s",
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
        "maxG11.dat-s"  ,
        "maxG51.dat-s"  ,
        # "maxG32.dat-s"  ,
        # "maxG55.dat-s"  ,
        # "maxG60.dat-s"  ,
    ]
end

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

if :RANDSDP in sets_to_test
    println("RANDSDP")
    _randsdp(optimizer, 0, 5, 5)
    for i in RANDSDP_TEST_SET
        @show i
        sol = _randsdp(optimizer, i, 5, 5)
        println2(FILE, "RANDSDP", "$i", sol)
    end
end
if :MIMO in sets_to_test
    println("MIMO")
    _mimo(optimizer, 0, 100)
    for n in MIMO_TEST_SET
        @show n
        sol = _mimo(optimizer, 0, n)
        println2(FILE, "MIMO", "$n", sol)
    end
end
if :SENSORLOC in sets_to_test
    println("SENSORLOC")
    for n in SENSORLOC_TEST_SET
        @show n
        sol = _sensorloc(optimizer, 0, n)
        println2(FILE, "SENSORLOC", "$n", sol)
    end
end
if :SDPLIB in sets_to_test
    println("gpp")
    for name in GPP_TEST_SET
        println(name)
        sol = _sdplib(optimizer, joinpath(datapath, name))
        println2(FILE, "SDPLIB_gp", name, sol)
    end
    println("max_cut")
    for name in MAXCUT_TEST_SET
        println(name)
        sol = _sdplib(optimizer, joinpath(datapath, name))
        println2(FILE, "SDPLIB_mc", name, sol)
    end
end


close(FILE)