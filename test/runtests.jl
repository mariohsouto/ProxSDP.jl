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
    # using SCS
    using Mosek
end

# MIMO ---------------------------------------------------------------
# include("mimo.jl")
# if Base.libblas_name == "libmkl_rt"
#     mimo(ProxSDPSolverInstance(), 0)
# else
#     # mimo(MosekSolver(), 0)
#     mimo(CSDPSolver(objtol=1e-4, maxiter=100000), 0)
# end
# println("-------------------------------\n")
# for i in 1:5
#     if Base.libblas_name == "libmkl_rt"
#         mimo(ProxSDPSolverInstance(), i)
#     else
#         # mimo(MosekSolver(), i)
#         mimo(CSDPSolver(objtol=1e-4, maxiter=100000), i)
#     end
# end

# include("rand_sdp.jl")
# if Base.libblas_name == "libmkl_rt"
#     rand_sdp(ProxSDPSolverInstance(), 0)
# else
#     rand_sdp(MosekSolver(), 0)
# end
# rand_sdp(ProxSDPSolverInstance(), 0)


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
