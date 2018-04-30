
path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__), "data")

using JuMP
using Base.Test
import Base.isempty

if Base.libblas_name == "libmkl_rt"
    using ProxSDP
    using MathOptInterface
     const MOI = MathOptInterface
     using MathOptInterfaceUtilities
     const MOIU = MathOptInterfaceUtilities
else
    # using CSDP 
    using SCS
    # using Mosek
end

# Quadratic knapsack
include("sensor_loc.jl")
for seed in 1:1
    if Base.libblas_name == "libmkl_rt"
        sensor_loc(ProxSDPSolverInstance(), seed)
    else
        # sensor_loc(CSDPSolver(maxiter=100000), seed)
        sensor_loc(SCSSolver(max_iters=1000000, eps=1e-5), seed)
        # sensor_loc(MosekSolver(), seed)
    end
end

# MIMO ---------------------------------------------------------------
# include("mimo.jl")
# SNR = [0.05]
# for snr in SNR
#     rank_list = []
#     error_list = []
#     time_list = []
#     for seed in 1:50
#         tic()
#         if Base.libblas_name == "libmkl_rt"
#             rank, decode_error = mimo(ProxSDPSolverInstance(), seed, snr)
#         else
#             # rank, decode_error = mimo(MosekSolver(), seed, snr)
#             rank, decode_error = mimo(CSDPSolver(maxiter=100000), seed, snr)
#             # rank, decode_error = mimo(SCSSolver(max_iters=1000000, eps=1e-4), seed, snr)
#         end
#         elapsed = toc()
#         push!(rank_list, rank)
#         push!(error_list, decode_error)
#         push!(time_list, elapsed)
#     end
#     println("\n:---------------------------------------:")
#     @show rank_list
#     @show error_list
#     @show mean(rank_list)
#     @show mean(error_list)
#     @show mean(time_list)
#     @show count_error = sum(error_list.>0.0)
# end

# Graph equipartition problem -----------------------------------------
include("sdplib.jl")
@testset "Graph" begin
    paths = String[]

    push!(paths, "data/gpp124-1.dat-s")
    push!(paths, "data/gpp124-1.dat-s")
    push!(paths, "data/gpp124-2.dat-s")
    push!(paths, "data/gpp124-3.dat-s")
    push!(paths, "data/gpp124-4.dat-s")
    push!(paths, "data/gpp250-1.dat-s")
    push!(paths, "data/gpp250-2.dat-s")
    push!(paths, "data/gpp250-3.dat-s")
    push!(paths, "data/gpp250-4.dat-s")
    push!(paths, "data/gpp500-1.dat-s")
    push!(paths, "data/gpp500-2.dat-s")
    push!(paths, "data/gpp500-3.dat-s")
    push!(paths, "data/gpp500-4.dat-s")
    push!(paths, "data/equalG11.dat-s")
    push!(paths, "data/equalG51.dat-s")

    for path in paths
        @show path
        if Base.libblas_name == "libmkl_rt"
            sdplib(ProxSDPSolverInstance(), path)
        else
            sdplib(CSDPSolver(objtol=1e-4, maxiter=100000), path)
            # sdplib(SCSSolver(max_iters=1000000, eps=1e-4), path)
            # sdplib(SCSSolver(eps=1e-4), path)
            # sdplib(MosekSolver(), path)
        end
    end
end

# Max-Cut problem -----------------------------------------
# include("sdplib.jl")
# @testset "Graph" begin
#     paths = String[]
#     push!(paths, "data/mcp250-1.dat-s")
#     push!(paths, "data/mcp250-1.dat-s")
#     push!(paths, "data/mcp250-2.dat-s")
#     push!(paths, "data/mcp250-3.dat-s")
#     push!(paths, "data/mcp250-4.dat-s")
#     push!(paths, "data/mcp500-1.dat-s")
#     push!(paths, "data/mcp500-2.dat-s")
#     push!(paths, "data/mcp500-3.dat-s")
#     push!(paths, "data/mcp500-4.dat-s")

#     for path in paths
#         @show path
#         if Base.libblas_name == "libmkl_rt"
#             sdplib(ProxSDPSolverInstance(), path)
#         else
#             sdplib(CSDPSolver(objtol=1e-4, maxiter=100000), path)
#             # sdplib(SCSSolver(max_iters=1000000, eps=1e-4), path)
#             # sdplib(SCSSolver(eps=1e-4), path)
#             # sdplib(MosekSolver(), path)
#         end
#     end
# end
