path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__), "data")

using JuMP
using Base.Test
import Base.isempty

# if Base.libblas_name == "libmkl_rt"
#     using ProxSDP
#     using MathOptInterface
#     const MOI = MathOptInterface
#     using MathOptInterfaceUtilities
#     const MOIU = MathOptInterfaceUtilities
# elseif VERSION < v"0.6.0"
#     using CSDP 
# else
#     using SCS
#     if is_linux()
#         using Mosek
#     end
# end

# include("jumptest.jl")
# include("max_cut.jl")

# tic()
# @testset "MIMO" begin
#     if Base.libblas_name == "libmkl_rt"
#         mimo(ProxSDPSolverInstance())
#     elseif VERSION < v"0.6.0"
#         mimo(CSDPSolver(objtol=1e-4, maxiter=10000, fastmode=1))
#     else
#         mimo(SCSSolver(max_iters=1000000, eps=1e-4))
#         # mimo(MosekSolver(MSK_IPAR_NUM_THREADS = 6))
#     end
# end
# toc()

# tic()
# @testset "Max-Cut" begin
#     # path = "data/mcp250-1.dat-s" 
#     path = "data/maxG55.dat-s" 
#     if Base.libblas_name == "libmkl_rt"
#         max_cut(ProxSDPSolverInstance(), path)
#     elseif VERSION < v"0.6.0"
#         max_cut(CSDPSolver(objtol=1e-4, maxiter=10000, fastmode=1), path)
#     else
#         # max_cut(SCSSolver(max_iters=1000000, eps=1e-4), path)
#         max_cut(MosekSolver(), path)
#     end
# end
# toc()

using ProxSDP
using MathOptInterface
const MOI = MathOptInterface
using MathOptInterfaceUtilities
const MOIU = MathOptInterfaceUtilities
include("jumptest.jl")
include("max_cut.jl")
mimo(ProxSDPSolverInstance())
# path = "data/mcp500-1.dat-s" 
# path = "data/maxG51.dat-s" 
# max_cut(ProxSDPSolverInstance(), path)
