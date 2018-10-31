path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__), "data")
# using JuMP
using Base.Test
# import Base.is_empty

use_MOI = true
# set_to_test = :MIMO
# set_to_test = :RANDSDP
# set_to_test = :SDPLIB
set_to_test = :SENSORLOC

@static if use_MOI#Base.libblas_name == "libmkl_rt"
    using ProxSDP, MathOptInterface
    include("moi_init.jl")
    # optimizer = MOIU.CachingOptimizer(ProxSDPModelData{Float64}(), ProxSDP.Optimizer(log_verbose=true, timer_verbose = true))
    optimizer = ProxSDP.Solver(log_verbose=true, timer_verbose = true)
else
    using JuMP
    # using CSDP
    # optimizer = CSDPSolver(objtol=1e-4, maxiter=100000)
    using SCS
    optimizer = SCSSolver(eps=1e-4, verbose=true)
    # using Mosek
    # optimizer = MosekSolver()
end

if use_MOI
    if set_to_test == :MIMO
        include("base_mimo.jl")
        include("moi_mimo.jl")
        # include("sensor_loc.jl")
        moi_mimo(optimizer, 0, 100)
        for n in [100, 500, 1000, 2000, 3000, 4000, 5000]
            @show n
            moi_mimo(optimizer, 0, n)
            # sensor_loc(optimizer, 0)
        end
    elseif set_to_test == :RANDSDP
        include("base_randsdp.jl")
        include("moi_randsdp.jl")
        moi_randsdp(optimizer, 0, 5, 5)
        for i in 1:1
            moi_randsdp(optimizer, i, 5, 5)
        end
    elseif set_to_test == :SDPLIB
        include("base_sdplib.jl")
        include("moi_sdplib.jl")
        moi_sdplib(optimizer, joinpath(datapath, "gpp124-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp124-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp124-2.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp124-3.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp124-4.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp250-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp250-2.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp250-3.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp250-4.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp500-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp500-2.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp500-3.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp500-4.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "equalG11.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "equalG51.dat-s"))

        println("max_cut")
        moi_sdplib(optimizer, joinpath(datapath, "mcp124-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp124-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp124-2.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp124-3.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp124-4.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp250-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp250-2.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp250-3.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp250-4.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp500-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp500-2.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp500-3.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp500-4.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "maxG11.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "maxG51.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "maxG32.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "maxG55.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "maxG60.dat-s"))

    elseif set_to_test == :SENSORLOC
        include("base_sensorloc.jl")
        include("moi_sensorloc.jl")
        for n in 50:50:300
            @show n
            moi_sensorloc(optimizer, 0, n)
        end
    end
else
    if set_to_test == :MIMO
        include("base_mimo.jl")
        include("jump_mimo.jl")
        jump_mimo(optimizer, 0, 10)

        for i in [100, 500, 1000, 2000, 3000, 4000, 5000]
            @show i
            tic()
            jump_mimo(optimizer, 0, i)
            toc()
        end
    elseif set_to_test == :RANDSDP
        include("base_randsdp.jl")
        include("jump_randsdp.jl")
        jump_randsdp(optimizer, 0, 5, 5)
        for i in 1:1
            jump_randsdp(optimizer, i, 5, 5)
        end
    elseif set_to_test == :SDPLIB
        include("base_sdplib.jl")
        include("jump_sdplib.jl")

        jump_sdplib(optimizer, joinpath(datapath, "gpp124-1.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp124-1.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp124-2.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp124-3.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp124-4.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp250-1.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp250-2.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp250-3.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp250-4.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp500-1.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp500-2.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp500-3.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "gpp500-4.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "equalG11.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "equalG51.dat-s"))

        println("max_cut")
        jump_sdplib(optimizer, joinpath(datapath, "mcp124-1.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp124-1.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp124-2.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp124-3.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp124-4.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp250-1.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp250-2.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp250-3.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp250-4.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp500-1.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp500-2.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp500-3.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "mcp500-4.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "maxG11.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "maxG51.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "maxG32.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "maxG55.dat-s"))
        jump_sdplib(optimizer, joinpath(datapath, "maxG60.dat-s"))
    elseif set_to_test == :SENSORLOC
        include("base_sensorloc.jl")
        include("jump_sensorloc.jl")
        jump_sensorloc(optimizer, 0, 10)
        for n in 50:50:300
            @show n
            jump_sensorloc(optimizer, 0, n)
        end
    end
end