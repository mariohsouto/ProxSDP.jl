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
# include("sdplib.jl")
include("mimo.jl")
if Base.libblas_name == "libmkl_rt"
    mimo(ProxSDPSolverInstance())
else
    mimo(MosekSolver())
end

# @testset "MIMO" begin
#     paths = String[]
#     # push!(paths, "data/mcp250-1.dat-s")
#     # push!(paths, "data/mcp250-1.dat-s")
#     # push!(paths, "data/mcp500-1.dat-s")
#     # push!(paths, "data/mcp500-4.dat-s")
#     # push!(paths, "data/maxG11.dat-s")
#     # push!(paths, "data/maxG51.dat-s")
#     # push!(paths, "data/maxG32.dat-s")
#     push!(paths, "data/theta1.dat-s")
#     push!(paths, "data/theta2.dat-s")
#     push!(paths, "data/theta3.dat-s")
#     push!(paths, "data/theta4.dat-s")
#     push!(paths, "data/theta5.dat-s")
#     push!(paths, "data/theta6.dat-s")
#     push!(paths, "data/thetaG11.dat-s")
#     push!(paths, "data/thetaG51.dat-s")
#     push!(paths, "data/truss1.dat-s")
#     push!(paths, "data/truss2.dat-s")
#     push!(paths, "data/truss3.dat-s")
#     push!(paths, "data/truss4.dat-s")
#     push!(paths, "data/truss5.dat-s")
#     push!(paths, "data/truss6.dat-s")
#     push!(paths, "data/truss7.dat-s")
#     # push!(paths, "data/truss8.dat-s")

#     for path in paths
#         sdplib(ProxSDPSolverInstance(), path)
#         @show path
#         # if Base.libblas_name == "libmkl_rt"
#         #     sdplib(ProxSDPSolverInstance(), path)
#         # elseif VERSION < v"0.6.0"
#         # else
#             # sdplib(CSDPSolver(objtol=1e-4, maxiter=10000, fastmode=1), path)
#             # sdplib(SCSSolver(max_iters=1000000, eps=1e-4), path)
#             # sdplib(SCSSolver(), path)
#             # max_cut(MosekSolver(), path)
#         end
#     end
# end
