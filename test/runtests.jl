
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

include("mimo.jl")
if Base.libblas_name == "libmkl_rt"
    mimo(ProxSDPSolverInstance(), 0)
    for i in 1:1
        mimo(ProxSDPSolverInstance(), i)
    end
else
    # mimo(MosekSolver(), 0)
    mimo(CSDPSolver(objtol=1e-4, maxiter=100000), 0)
    # mimo(SCSSolver(eps=1e-4), 0)
    for i in 1:1
        mimo(CSDPSolver(objtol=1e-4, maxiter=100000), i)
        # mimo(SCSSolver(eps=1e-4), i)
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
