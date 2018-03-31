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
    # using CSDP 
    # using SCS
    using Mosek
end

# include("max_cut.jl")
include("sdplib.jl")

@testset "MIMO" begin
    paths = String[]
    push!(paths, "data/truss1.dat-s")

    for path in paths
        @show path
        if Base.libblas_name == "libmkl_rt"
            sdplib(ProxSDPSolverInstance(), path)
        # elseif VERSION < v"0.6.0"
        else
            # sdplib(CSDPSolver(objtol=1e-4, maxiter=10000, fastmode=1), path)
            # sdplib(SCSSolver(max_iters=1000000, eps=1e-4), path)
            # sdplib(SCSSolver(), path)
            max_cut(MosekSolver(), path)
        end
    end
end
