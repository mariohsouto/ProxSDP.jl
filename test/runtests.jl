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
    # using SCS
    # using Mosek
end

# include("mimo.jl")
# mimo(ProxSDPSolverInstance())

include("sdplib.jl")

@testset "MIMO" begin
    paths = String[]

    # Graph equipartition problem
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

    # Truss topology
    # push!(paths, "data/arch0.dat-s")
    # push!(paths, "data/arch0.dat-s")
    # push!(paths, "data/arch2.dat-s")
    # push!(paths, "data/arch2.dat-s")
    # push!(paths, "data/arch4.dat-s")
    # push!(paths, "data/arch8.dat-s")

    # Max-Cut
    push!(paths, "data/mcp250-1.dat-s")
    push!(paths, "data/mcp250-1.dat-s")
    push!(paths, "data/mcp250-2.dat-s")
    push!(paths, "data/mcp250-3.dat-s")
    push!(paths, "data/mcp250-4.dat-s")
    push!(paths, "data/mcp500-1.dat-s")
    push!(paths, "data/mcp500-2.dat-s")
    push!(paths, "data/mcp500-3.dat-s")
    push!(paths, "data/mcp500-4.dat-s")
    push!(paths, "data/maxG11.dat-s")
    push!(paths, "data/maxG51.dat-s")
    push!(paths, "data/maxG32.dat-s")

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
