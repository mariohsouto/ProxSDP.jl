path = joinpath(dirname(@__FILE__),"..","..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__),"data")

using JuMP
using Base.Test
import Base.isempty

if VERSION > v"0.6.0"
    using ProxSDP
    using MathOptInterface
    const MOI = MathOptInterface
    using MathOptInterfaceUtilities
    const MOIU = MathOptInterfaceUtilities
    # using Mosek
    # using MathOptInterfaceMosek
else
    using CSDP
    using SCS
end

# ProxSDP.runpsdp(datapath)

include("jumptest.jl")
include("max_cut.jl")

@testset "MIMO" begin
    if VERSION > v"0.6.0"
        mimo(ProxSDPSolverInstance())
        # mimo(MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10000))
    else
        mimo(SCSSolver(max_iters=1000000, eps=1e-4))
        mimo(CSDPSolver(objtol=1e-4, maxiter=10000, fastmode=1))
    end
end

@testset "Max-Cut" begin
    path = "data/mcp500-1.dat-s" 
    if VERSION > v"0.6.0"
        max_cut(ProxSDPSolverInstance(), path)
        # mimo(MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10000))
    else
        max_cut(SCSSolver(max_iters=1000000, eps=1e-4), path)
        max_cut(CSDPSolver(objtol=1e-4, maxiter=10000, fastmode=1), path)
    end
end
