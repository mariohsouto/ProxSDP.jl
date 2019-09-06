
struct CircularVector{T}
    v::Vector{T}
    l::Int
    CircularVector{T}(l::Integer) where T = new(zeros(T, l), l)
end

function Base.getindex(V::CircularVector{T}, i::Int) where T
    return V.v[mod1(i, V.l)]
end

function Base.setindex!(V::CircularVector{T}, val::T, i::Int) where T
    V.v[mod1(i, V.l)] = val
end

Base.@kwdef mutable struct Options

    # Printing options
    log_verbose::Bool = false
    log_freq::Int = 100
    timer_verbose::Bool = false
    timer_file::Bool = false

    # time options
    time_limit::Float64 = 3600_00. #100 hours

    # Default tolerances
    tol_primal::Float64 = 1e-3
    tol_dual::Float64 = 1e-3
    tol_psd::Float64 = 1e-6
    tol_soc::Float64 = 1e-6

    # Bounds on beta (dual_step / primal_step) [larger bounds may lead to numerical inaccuracy]
    min_beta::Float64 = 1e-5
    max_beta::Float64 = 1e+5
    initial_beta::Float64 = 1.

    # Adaptive primal-dual steps parameters [adapt_decay above .7 may lead to slower convergence]
    initial_adapt_level::Float64 = .9
    adapt_decay::Float64 = .8
    adapt_window::Int64 = 50

    # PDHG parameters
    convergence_window::Int = 200
    convergence_check::Int = 50
    max_iter::Int = Int(1e+5)

    # Linesearch parameters
    max_linsearch_steps::Int = 2000
    delta::Float64 = .999
    initial_theta::Float64 = 1.
    linsearch_decay::Float64 = .9

    # Spectral decomposition parameters
    full_eig_decomp::Bool = false
    max_target_rank_krylov_eigs::Int = 16
    min_size_krylov_eigs::Int = 100
    warm_start_eig::Bool = true

    # Reduce rank [warning: heuristics]
    reduce_rank::Bool = false
    rank_slack::Int = 3

    # equilibration parameters
    equilibration::Bool = false
    equilibration_iters::Int = 100
    equilibration_lb::Float64 = -10.0
    equilibration_ub::Float64 = +10.0
    equilibration_limit::Float64 = 0.8
    equilibration_force::Bool = false

    # spectral norm [using exact norm via svds may result in nondeterministic behavior]
    approx_norm::Bool = true
end

function Options(args)
    options = Options()
    parse_args!(options, args)
    return options
end

function parse_args!(options, args)
    for i in args
        parse_arg!(options, i)
    end
    return nothing
end

function parse_arg!(options::Options, arg)
    fields = fieldnames(Options)
    name = arg[1]
    value = arg[2]
    if name in fields
        setfield!(options, name, value)
    end
    return nothing
end

mutable struct AffineSets
    n::Int  # Size of primal variables
    p::Int  # Number of linear equalities
    m::Int  # Number of linear inequalities
    extra::Int  # Number of adition linear equalities (for disjoint cones)
    A::SparseMatrixCSC{Float64,Int64} 
    G::SparseMatrixCSC{Float64,Int64}
    b::Vector{Float64}
    h::Vector{Float64}
    c::Vector{Float64}
end

mutable struct SDPSet
    vec_i::Vector{Int}
    mat_i::Vector{Int}
    tri_len::Int
    sq_len::Int
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

struct CPResult
    status::Int
    status_string::String
    primal::Vector{Float64}
    dual::Vector{Float64}
    slack::Vector{Float64}
    primal_residual::Float64
    dual_residual::Float64
    objval::Float64
    dual_objval::Float64
    gap::Float64
    time::Float64
    final_rank::Int
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

mutable struct Residuals
    dual_gap::Float64
    prim_obj::Float64
    dual_obj::Float64
    equa_feasibility::Float64 
    ineq_feasibility::Float64
    feasibility::Float64
    primal_residual::CircularVector{Float64}
    dual_residual::CircularVector{Float64}
    comb_residual::CircularVector{Float64}

    Residuals(window::Int) = new(
        .0, .0, .0, .0, .0, .0,
        CircularVector{Float64}(2*window),
        CircularVector{Float64}(2*window),
        CircularVector{Float64}(2*window)
    )
end

const ViewVector = SubArray#{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
const ViewScalar = SubArray#{Float64, 1, Vector{Float64}, Tuple{Int}, true}

mutable struct AuxiliaryData
    m::Vector{Symmetric{Float64,Matrix{Float64}}}

    Mty::Vector{Float64}
    Mty_old::Vector{Float64}

    Mx::Vector{Float64}
    Mx_old::Vector{Float64}
    residual::Vector{Float64}

    y_half::Vector{Float64}
    y_temp::Vector{Float64}

    soc_v::Vector{ViewVector}
    soc_s::Vector{ViewScalar}

    function AuxiliaryData(aff::AffineSets, cones::ConicSets) 
        Mx_old = zeros(aff.p+aff.m)
    new(
        [Symmetric(zeros(sdp.sq_side, sdp.sq_side), :L) for sdp in cones.sdpcone], 
        zeros(aff.n), zeros(aff.n),
        zeros(aff.p+aff.m), Mx_old, zeros(aff.p+aff.m), 
        zeros(aff.p+aff.m), zeros(aff.p+aff.m),
        ViewVector[], ViewScalar[]
)
    end
end

mutable struct Matrices
    M::SparseMatrixCSC{Float64,Int64}
    Mt::SparseMatrixCSC{Float64,Int64}
    c::Vector{Float64}

    Matrices(M, Mt, c) = new(M, Mt, c)
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

    Params() = new()
end
