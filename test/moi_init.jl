const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

sympackedlen(n) = div(n*(n+1), 2)
sympackeddim(n) = div(isqrt(1+8n) - 1, 2)
function ivech!(out::AbstractMatrix{T}, v::AbstractVector{T}) where T
    n = sympackeddim(length(v))
    n1, n2 = size(out)
    @assert n == n1 == n2
    c = 0
    for j in 1:n, i in 1:j
        c += 1
        out[i,j] = v[c]
    end
    return out
end
function ivech(v::AbstractVector{T}) where T
    n = sympackeddim(length(v))
    out = zeros(n, n)
    ivech!(out, v)
    
    return out
end

solvers = Tuple{String, Any}[]

#MOIU.@model ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)

#=
    CSDP & SDPA
=#

# ]activate soi
# using CSDP
# using SDPA

# MOIU.@model(SOI_ModelData,
#             (),
#             (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan),
#             (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
#              MOI.PositiveSemidefiniteConeTriangle),
#             (),
#             (MOI.SingleVariable,),
#             (MOI.ScalarAffineFunction,),
#             (MOI.VectorOfVariables,),
#             (MOI.VectorAffineFunction,))

# const CSDP_optimizer = 
# MOIB.full_bridge_optimizer(
#     MOIU.CachingOptimizer(
#         SOI_ModelData{Float64}(), CSDP.Optimizer()),#printlevel=0))
#     Float64
#     )

# push!(solvers, ("CSDP", CSDP_optimizer)) # eps = ???

# const SDPA_optimizer = 
# MOIB.full_bridge_optimizer(
#     MOIU.CachingOptimizer(
#         SOI_ModelData{Float64}(), SDPA.Optimizer()),#printlevel=0))
#     Float64
#     )

# push!(solvers, ("SDPA", SDPA_optimizer)) # eps = ???

#=
    COSMO
=#

# using COSMO

# MOIU.@model(COSMOModelData,
#         (),
#         (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
#         (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.SecondOrderCone,
#          MOI.PositiveSemidefiniteConeSquare, MOI.PositiveSemidefiniteConeTriangle, MOI.ExponentialCone, MOI.DualExponentialCone),
#         (MOI.PowerCone, MOI.DualPowerCone),
#         (MOI.SingleVariable,),
#         (MOI.ScalarAffineFunction, MOI.ScalarQuadraticFunction),
#         (MOI.VectorOfVariables,),
#         (MOI.VectorAffineFunction,),);

# const COSMO_optimizer = MOIU.CachingOptimizer(MOIU.UniversalFallback(COSMOModelData{Float64}()),
#     COSMO.Optimizer(eps_abs = 1e-4, eps_rel = 1e-4 ));

# push!(solvers, ("COSMO", COSMO_optimizer))

#=
    SeDuMi
=#

import SeDuMi
const optimizer = SeDuMi.Optimizer(fid=0)

MOIU.@model(SeDuMi_ModelData, (), (),
            (MOI.Zeros, MOI.Nonnegatives, MOI.SecondOrderCone,
             MOI.RotatedSecondOrderCone, MOI.PositiveSemidefiniteConeTriangle),
            (), (), (), (MOI.VectorOfVariables,), (MOI.VectorAffineFunction,))

const SeDuMi_optimizer = MOIB.full_bridge_optimizer(MOIU.CachingOptimizer(
    MOIU.UniversalFallback(SeDuMi_ModelData{Float64}()),
    SeDuMi.Optimizer(fid=0)),
    Float64)
    
push!(solvers, ("SeDuMi", SeDuMi_optimizer))
