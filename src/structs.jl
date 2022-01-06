
struct CircularVector{T}
    v::Vector{T}
    l::Int
    CircularVector{T}(l::Integer) where T = new(zeros(T, l), l)
end
function min_abs_diff(v::CircularVector{T}) where T
    val = Inf
    for i in 1:Base.length(v)
        val = min(val, abs(v[i] - v[i-1]))
    end
    return val
end
function max_abs_diff(v::CircularVector{T}) where T
    val = 0.0
    for i in 1:Base.length(v)
        val = max(val, abs(v[i] - v[i-1]))
    end
    return val
end

function Base.getindex(V::CircularVector{T}, i::Int) where T
    return V.v[mod1(i, V.l)]
end
function Base.setindex!(V::CircularVector{T}, val::T, i::Int) where T
    V.v[mod1(i, V.l)] = val
end
function Base.length(V::CircularVector{T}) where T
    return V.l
end

mutable struct AffineSets
    n::Int  # Size of primal variables
    p::Int  # Number of linear equalities
    m::Int  # Number of linear inequalities
    extra::Int  # Number of adition linear equalities (for disjoint cones)
    A::SparseArrays.SparseMatrixCSC{Float64,Int64}
    G::SparseArrays.SparseMatrixCSC{Float64,Int64}
    b::Vector{Float64}
    h::Vector{Float64}
    c::Vector{Float64}
end

mutable struct SDPSet
    vec_i::Vector{Int}
    mat_i::Vector{Int}
    tri_len::Int
    sq_side::Int
end

mutable struct SOCSet
    idx::Vector{Int}
    len::Int
end

mutable struct ConicSets
    sdpcone::Vector{SDPSet}
    socone::Vector{SOCSet}
end

Base.@kwdef mutable struct Result
    status::Int = 0
    status_string::String = "Problem not solved"
    primal::Vector{Float64} = Float64[]
    dual_cone::Vector{Float64} = Float64[]
    dual_eq::Vector{Float64} = Float64[]
    dual_in::Vector{Float64} = Float64[]
    slack_eq::Vector{Float64} = Float64[]
    slack_in::Vector{Float64} = Float64[]
    primal_residual::Float64 = NaN
    dual_residual::Float64 = NaN
    objval::Float64 = NaN
    dual_objval::Float64 = NaN
    gap::Float64 = NaN
    time::Float64 = NaN
    iter::Int = -1
    final_rank::Int = -1
    primal_feasible_user_tol::Bool = false
    dual_feasible_user_tol::Bool = false
    certificate_found::Bool = false
    result_count::Int = 0
end

mutable struct PrimalDual
    x::Vector{Float64}
    x_old::Vector{Float64}
    y::Vector{Float64}
    y_old::Vector{Float64}

    PrimalDual(aff::AffineSets) = new(
        zeros(aff.n), zeros(aff.n), zeros(aff.m+aff.p), zeros(aff.m+aff.p)
    )
end

mutable struct WarmStart
    x::Vector{Float64}
    y_eq::Vector{Float64}
    y_in::Vector{Float64}
end

mutable struct Residuals
    dual_gap::CircularVector{Float64}
    prim_obj::CircularVector{Float64}
    dual_obj::CircularVector{Float64}
    equa_feasibility::Float64 
    ineq_feasibility::Float64
    feasibility::CircularVector{Float64}
    primal_residual::CircularVector{Float64}
    dual_residual::CircularVector{Float64}
    comb_residual::CircularVector{Float64}

    Residuals(window::Int) = new(
        CircularVector{Float64}(2*window),
        CircularVector{Float64}(2*window),
        CircularVector{Float64}(2*window),
        .0,
        .0,
        CircularVector{Float64}(2*window),
        CircularVector{Float64}(2*window),
        CircularVector{Float64}(2*window),
        CircularVector{Float64}(2*window)
    )
end

const ViewVector = SubArray#{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
const ViewScalar = SubArray#{Float64, 1, Vector{Float64}, Tuple{Int}, true}

mutable struct AuxiliaryData
    m::Vector{LinearAlgebra.Symmetric{Float64,Matrix{Float64}}}

    Mty::Vector{Float64}
    Mty_old::Vector{Float64}

    Mx::Vector{Float64}
    Mx_old::Vector{Float64}

    y_half::Vector{Float64}
    y_temp::Vector{Float64}

    soc_v::Vector{ViewVector}
    soc_s::Vector{ViewScalar}

    function AuxiliaryData(aff::AffineSets, cones::ConicSets) 
        new(
            [LinearAlgebra.Symmetric(zeros(sdp.sq_side, sdp.sq_side), :U) for sdp in cones.sdpcone], 
            zeros(aff.n), zeros(aff.n),
            zeros(aff.p+aff.m), zeros(aff.p+aff.m),
            zeros(aff.p+aff.m), zeros(aff.p+aff.m),
            ViewVector[], ViewScalar[]
        )
    end
end

mutable struct Matrices
    M::SparseArrays.SparseMatrixCSC{Float64,Int64}
    Mt::SparseArrays.SparseMatrixCSC{Float64,Int64}
    c::Vector{Float64}
end

mutable struct Params
    current_rank::Vector{Int}
    target_rank::Vector{Int}
    rank_update::Int
    update_cont::Int
    min_eig::Vector{Float64}
    iter::Int
    stop_reason::Int
    stop_reason_string::String
    iteration::Int
    primal_step::Float64
    primal_step_old::Float64
    dual_step::Float64
    theta::Float64
    beta::Float64
    adapt_level::Float64
    window::Int
    time0::Float64
    norm_c::Float64
    norm_b::Float64
    norm_h::Float64
    sqrt2::Float64

    dual_feasibility::Float64
    dual_feasibility_check::Bool

    certificate_search::Bool
    certificate_search_min_iter::Int
    certificate_found::Bool

    # solution backup

    Params() = new()
end
