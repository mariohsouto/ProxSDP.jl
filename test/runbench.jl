
path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__), "data")

# using JuMP
using Base.Test
# import Base.isempty

use_MOI = true
# set_to_test = :MIMO
# set_to_test = :RANDSDP
set_to_test = :SDPLIB
# set_to_test = :SENSORLOC

@static if use_MOI#Base.libblas_name == "libmkl_rt"
    using ProxSDP, MathOptInterface
    include("moi_init.jl")
    optimizer = MOIU.CachingOptimizer(ProxSDPModelData{Float64}(), ProxSDPOptimizer())
else
    using JuMP
    using CSDP
    optimizer = CSDPSolver(objtol=1e-4, maxiter=100000)
    # using SCS
    # optimizer = SCSSolver(eps=1e-4)
    # using Mosek
    # optimizer = MosekSolver()
end


if use_MOI
    if set_to_test == :MIMO
        include("base_mimo.jl")
        include("moi_mimo.jl")
        # include("sensor_loc.jl")
        moi_mimo(optimizer, 0, 10)
        for n in [10,10]
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
        # moi_sdplib(optimizer, joinpath(datapath, "gpp124-1.dat-s"))
        # moi_sdplib(optimizer, joinpath(datapath, "gpp124-1.dat-s"))
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
        moi_sdplib(optimizer, joinpath(datapath, "mcp250-1.dat-s"))
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
        # moi_sdplib(optimizer, joinpath(datapath, "maxG32.dat-s"))
        # for i in 1:1
        #     moi_sdplib(optimizer, joinpath(datapath, "gpp124-1.dat-s"))
        # end
    elseif set_to_test == :SENSORLOC
        include("base_sensorloc.jl")
        include("moi_sensorloc.jl")
        moi_sensorloc(optimizer, 123, 10)
    end
else
    if set_to_test == :MIMO
        include("base_mimo.jl")
        include("jump_mimo.jl")
        jump_mimo(optimizer, 0, 5)
        for i in 1:1
            jump_mimo(optimizer, i, 5)
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
        for i in 1:1
            jump_sdplib(optimizer, joinpath(datapath, "gpp124-1.dat-s"))
        end
    elseif set_to_test == :SENSORLOC
        include("base_sensorloc.jl")
        include("jump_sensorloc.jl")
        jump_sensorloc(optimizer, 123, 10)
    end
end

# include("sdplib.jl")
# @testset "MIMO" begin
#     paths = String[]

#     # Graph equipartition problem
#     # push!(paths, "data/gpp124-1.dat-s")
#     # push!(paths, "data/gpp124-1.dat-s")
#     # push!(paths, "data/gpp124-2.dat-s")
#     # push!(paths, "data/gpp124-3.dat-s")
#     # push!(paths, "data/gpp124-4.dat-s")
#     # push!(paths, "data/gpp250-1.dat-s")
#     # push!(paths, "data/gpp250-2.dat-s")
#     # push!(paths, "data/gpp250-3.dat-s")
#     # push!(paths, "data/gpp250-4.dat-s")
#     # push!(paths, "data/gpp500-1.dat-s")
#     # push!(paths, "data/gpp500-2.dat-s")
#     # push!(paths, "data/gpp500-3.dat-s")
#     # push!(paths, "data/gpp500-4.dat-s")
#     # push!(paths, "data/equalG11.dat-s")
#     # push!(paths, "data/equalG51.dat-s")

#     # Truss topology
#     push!(paths, "data/arch0.dat-s")
#     push!(paths, "data/arch0.dat-s")
#     push!(paths, "data/arch2.dat-s")
#     push!(paths, "data/arch2.dat-s")
#     push!(paths, "data/arch4.dat-s")
#     push!(paths, "data/arch8.dat-s")

#     # Max-Cut
#     # push!(paths, "data/mcp250-1.dat-s")
#     # push!(paths, "data/mcp250-1.dat-s")
#     # push!(paths, "data/mcp250-2.dat-s")
#     # push!(paths, "data/mcp250-3.dat-s")
#     # push!(paths, "data/mcp250-4.dat-s")
#     # push!(paths, "data/mcp500-1.dat-s")
#     # push!(paths, "data/mcp500-2.dat-s")
#     # push!(paths, "data/mcp500-3.dat-s")
#     # push!(paths, "data/mcp500-4.dat-s")
#     # push!(paths, "data/maxG11.dat-s")
#     # push!(paths, "data/maxG51.dat-s")
#     # push!(paths, "data/maxG32.dat-s")

#     # push!(paths, "data/truss1.dat-s")
#     # push!(paths, "data/truss2.dat-s")
#     # push!(paths, "data/truss3.dat-s")
#     # push!(paths, "data/truss4.dat-s")
#     # push!(paths, "data/truss5.dat-s")
#     # push!(paths, "data/truss6.dat-s")
#     # push!(paths, "data/truss7.dat-s")

#     for path in paths
#         @show path
#         if Base.libblas_name == "libmkl_rt"
#             sdplib(ProxSDPSolverInstance(), path)
#         else
#             # sdplib(CSDPSolver(objtol=1e-4, maxiter=100000), path)
#             # sdplib(SCSSolver(max_iters=1000000, eps=1e-4), path)
#             # sdplib(SCSSolver(eps=1e-4), path)
#             sdplib(MosekSolver(), path)
#         end
#     end
# end


# include("sensor_loc.jl")
# for seed in 1:1
#     if Base.libblas_name == "libmkl_rt"
#         sensor_loc(ProxSDPSolverInstance(), seed)
#     else
#         # sensor_loc(CSDPSolver(maxiter=100000), seed)
#         sensor_loc(SCSSolver(max_iters=1000000, eps=1e-5), seed)
#         # sensor_loc(MosekSolver(), seed)
#     end
# end

# include("base.jl")
# if Base.libblas_name == "libmkl_rt"
#     base_sdp(ProxSDPSolverInstance(), 0)
#     base_sdp2(ProxSDPSolverInstance(), 0)
# else
#     base_sdp(MosekSolver(), 0)
#     base_sdp2(MosekSolver(), 0)
#     # rand_sdp(CSDPSolver(objtol=1e-4, maxiter=100000), 0)
#     # rand_sdp(SCSSolver(eps=1e-4, max_iters=100000), 0)
# end

# include("rand_sdp.jl")
# if Base.libblas_name == "libmkl_rt"
#     rand_sdp(ProxSDPSolverInstance(), 0)
# else
#     rand_sdp(MosekSolver(), 0)
#     # rand_sdp(CSDPSolver(objtol=1e-4, maxiter=100000), 0)
#     # rand_sdp(SCSSolver(eps=1e-4, max_iters=100000), 0)
# end

# rand_sdp(MosekSolver(), 0)
