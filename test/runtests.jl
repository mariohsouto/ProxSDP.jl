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
 #elseif VERSION < v"0.6.0"
    
else
    using CSDP 
    using SCS
    using Mosek
end

#  include("jumptest.jl")
include("max_cut.jl")

# @testset "MIMO" begin
#     if Base.libblas_name == "libmkl_rt"
#         mimo(ProxSDPSolverInstance(),100)
#     #elseif VERSION < v"0.6.0"
#     else
#         mimo(CSDPSolver(objtol=1e-4, maxiter=10000, fastmode=1),100)
#         mimo(SCSSolver(max_iters=1000000),100)
#         mimo(MosekSolver())
#     end
# end

# tic()
@testset "Max-Cut" begin
    paths = String[]
    push!(paths, "data/mcp250-1.dat-s") 
    push!(paths, "data/mcp250-1.dat-s") 
    push!(paths, "data/mcp500-4.dat-s") #
    push!(paths, "data/mcp500-1.dat-s") 
    
    push!(paths, "data/maxG11.dat-s") 
    push!(paths, "data/maxG51.dat-s") 
    push!(paths, "data/maxG32.dat-s") 
    push!(paths, "data/maxG55.dat-s") 
    # path = "data/maxG55.dat-s" 

    for path in paths
        @show path
        @show path
        @show path
        @show path
        if Base.libblas_name == "libmkl_rt"
            max_cut(ProxSDPSolverInstance(), path)
        elseif VERSION < v"0.6.0"
        else
            # max_cut(CSDPSolver(objtol=1e-4, maxiter=10000, fastmode=1), path)
            # max_cut(SCSSolver(max_iters=1000000, eps=1e-4), path)
            max_cut(SCSSolver(), path)
            # max_cut(MosekSolver(), path)
        end
    end
end
