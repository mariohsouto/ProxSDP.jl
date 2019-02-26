
module ProxSDP

using MathOptInterface, TimerOutputs
using Arpack
using Compat
using Printf

include("MOI_wrapper.jl")
include("eigsolver.jl")


MOIU.@model _ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)

Solver(;args...) = MOIU.CachingOptimizer(_ProxSDPModelData{Float64}(), ProxSDP.Optimizer(args))

function get_solution(opt::MOIU.CachingOptimizer{Optimizer,T}) where T
    return opt.optimizer.sol
end

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

# --------------------------------
mutable struct Options
    log_verbose::Bool
    log_freq::Int
    timer_verbose::Bool
    max_iter::Int
    tol_primal::Float64
    tol_dual::Float64
    tol_eig::Float64
    tol_soc::Float64

    initial_theta::Float64
    initial_beta::Float64
    min_beta::Float64
    max_beta::Float64
    initial_adapt_level::Float64
    adapt_decay::Float64 # Rate the adaptivity decreases over time
    convergence_window::Int

    convergence_check::Int

    residual_relative_diff::Float64

    max_linsearch_steps::Int

    full_eig_decomp::Bool
    max_target_rank_krylov_eigs::Int
    min_size_krylov_eigs::Int

    function Options()
        opt = new()

        opt.log_verbose = false
        opt.log_freq = 100
        opt.timer_verbose = false

        opt.max_iter = Int(1e+5)

        opt.tol_primal = 1e-4
        opt.tol_dual = 1e-4
        opt.tol_eig = 1e-6
        opt.tol_soc = 1e-6

        opt.initial_theta = 1.0
        opt.initial_beta = 1.0
        opt.min_beta = 1e-7
        opt.max_beta = 1e+7
        opt.initial_adapt_level = 0.9
        opt.adapt_decay = 0.95
        opt.convergence_window = 100

        opt.convergence_check = 50

        opt.residual_relative_diff = 50.0

        opt.max_linsearch_steps = 1000

        opt.full_eig_decomp = false

        opt.max_target_rank_krylov_eigs = 16
        opt.min_size_krylov_eigs = 100

        return opt
    end
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
    A::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
    G::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
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
    primal::Vector{Float64}
    dual::Vector{Float64}
    slack::Vector{Float64}
    primal_residual::Float64
    dual_residual::Float64
    objval::Float64
    dual_objval::Float64
    gap::Float64
    time::Float64
end

mutable struct PrimalDual
    x::Vector{Float64}
    x_old::Vector{Float64}

    y::Vector{Float64}
    y_old::Vector{Float64}
    y_aux::Vector{Float64}

    PrimalDual(aff) = new(
        zeros(aff.n), zeros(aff.n), zeros(aff.m+aff.p), zeros(aff.m+aff.p), zeros(aff.m+aff.p)
    )
end

const ViewVector = SubArray#{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
const ViewScalar = SubArray#{Float64, 1, Vector{Float64}, Tuple{Int}, true}

mutable struct AuxiliaryData
    m::Vector{Symmetric{Float64,Matrix{Float64}}}
    Mty::Vector{Float64}
    Mty_old::Vector{Float64}
    Mty_aux::Vector{Float64}
    Mx::Vector{Float64}
    Mx_old::Vector{Float64}
    y_half::Vector{Float64}
    y_temp::Vector{Float64}
    MtMx::Vector{Float64}
    MtMx_old::Vector{Float64}
    Mtrhs::Vector{Float64}
    soc_v::Vector{ViewVector}
    soc_s::Vector{ViewScalar}
    function AuxiliaryData(aff::AffineSets, cones::ConicSets) 
        new([Symmetric(zeros(sdp.sq_side, sdp.sq_side), :L) for sdp in cones.sdpcone], zeros(aff.n), zeros(aff.n),
        zeros(aff.n), zeros(aff.p+aff.m), zeros(aff.p+aff.m), zeros(aff.p+aff.m), 
        zeros(aff.p+aff.m), zeros(aff.n), zeros(aff.n), zeros(aff.n),
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

    target_rank::Vector{Int}
    rank_update::Int
    update_cont::Int
    min_eig::Vector{Float64}

    iter::Int
    converged::Bool
    iteration::Int

    primal_step::Float64
    primal_step_old::Float64
    dual_step::Float64

    theta::Float64
    beta::Float64
    adapt_level::Float64

    window::Int

    # constants
    time0::Float64
    norm_c::Float64
    norm_rhs::Float64
    sqrt2::Float64

    Params() = new()
end

function printheader()
    println("======================================================================")
    println("          ProxSDP : Proximal Semidefinite Programming Solver          ")
    println("                 (c) Mario Souto and Joaquim D. Garcia, 2018          ")
    println("                                                Beta version          ")
    println("----------------------------------------------------------------------")
    println(" Initializing Primal-Dual Hybrid Gradient method                      ")
    println("----------------------------------------------------------------------")
    println("|  iter  | comb. res | prim. res |  dual res |    rank   |  time (s) |")
    println("----------------------------------------------------------------------")
end

function norm_scaling(affine_sets::AffineSets, cones::ConicSets)
    cte = (sqrt(2.0) / 2.0)
    rows = rowvals(affine_sets.A)
    cont = 1
    for sdp in cones.sdpcone, j in 1:sdp.sq_side, i in j:sdp.sq_side
        if i != j
            for line in nzrange(affine_sets.A, cont)
                affine_sets.A[rows[line], cont] *= cte
            end
        end
        cont += 1
    end
    rows = rowvals(affine_sets.G)
    cont = 1
    for sdp in cones.sdpcone, j in 1:sdp.sq_side, i in j:sdp.sq_side
        if i != j
            for line in nzrange(affine_sets.G, cont)
                affine_sets.G[rows[line], cont] *= cte
            end
        end
        cont += 1
    end
    cont = 1
    @inbounds for sdp in cones.sdpcone, j in 1:sdp.sq_side, i in j:sdp.sq_side
        if i != j
            affine_sets.c[cont] *= cte
        end
        cont += 1
    end
    return nothing
end

function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, opt)::CPResult

    p = Params()

    p.theta = opt.initial_theta             # Overrelaxation parameter
    p.adapt_level = opt.initial_adapt_level # Factor by which the stepsizes will be balanced 
    p.window = opt.convergence_window       # Convergence check window
    p.beta = opt.initial_beta

    p.time0 = time()

    p.norm_rhs = norm(vcat(affine_sets.b, affine_sets.h))
    p.norm_c = norm(affine_sets.c)

    p.rank_update, p.converged, p.update_cont = 0, false, 0
    p.target_rank = 2*ones(length(conic_sets.sdpcone))
    p.min_eig = zeros(length(conic_sets.sdpcone))

    analysis = false
    if analysis
        p.window = 1
    end

    if opt.log_verbose
        printheader()
    end

    @timeit "Init" begin
        # Scale objective function
        c_orig, var_ordering = preprocess!(affine_sets, conic_sets)
        A_orig, b_orig = copy(affine_sets.A), copy(affine_sets.b)
        G_orig, h_orig = copy(affine_sets.G), copy(affine_sets.h)
        rhs_orig = vcat(affine_sets.b, affine_sets.h)
        norm_scaling(affine_sets, conic_sets)
        # Initialization
        pair = PrimalDual(affine_sets)
        a = AuxiliaryData(affine_sets, conic_sets)
        arc = [ARPACKAlloc(Float64, 1) for i in conic_sets.sdpcone]
        map_socs!(pair.x, conic_sets, a)

        primal_residual = CircularVector{Float64}(2*p.window)
        dual_residual = CircularVector{Float64}(2*p.window)
        comb_residual = CircularVector{Float64}(2*p.window)

        # Diagonal scaling
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'
        rhs = vcat(affine_sets.b, affine_sets.h)
        mat = Matrices(M, Mt, affine_sets.c)
        mul!(a.Mtrhs, mat.Mt, rhs)
        
        # Stepsize parameters and linesearch parameters
        if minimum(size(M)) >= 2
            p.primal_step = 1.0 / Arpack.svds(M, nsv = 1)[1].S[1] #TODO review efficiency
        else
            p.primal_step = 1.0 / maximum(LinearAlgebra.svd(Matrix(M)).S) #TODO review efficiency
        end
        p.primal_step_old = p.primal_step
        p.dual_step = p.primal_step
        pair.x[1] = 1.0
    end

    # Fixed-point loop
    @timeit "CP loop" for k in 1:opt.max_iter

        p.iter = k

        # Primal update
        @timeit "primal" primal_step!(pair, a, conic_sets, mat, arc, opt, p)
        # Linesearch
        linesearch!(pair, a, affine_sets, mat, opt, p)
        # Compute residuals and update old iterates
        @timeit "residual" compute_residual!(pair, a, primal_residual, dual_residual, comb_residual, mat, p)
        # Print progress
        if opt.log_verbose && mod(k, opt.log_freq) == 0
            print_progress(primal_residual[k], dual_residual[k], p)
        end

        # Check convergence of inexact fixed-point
        p.rank_update += 1
        if primal_residual[k] < opt.tol_primal && dual_residual[k] < opt.tol_dual && k > opt.convergence_check
            if convergedrank(p, conic_sets, opt) && soc_convergence(a, conic_sets, pair, opt, p)
                p.converged = true
                best_prim_residual, best_dual_residual = primal_residual[k], dual_residual[k]
                if opt.log_verbose
                    print_progress(primal_residual[k], dual_residual[k], p)
                end
                break
            elseif p.rank_update > p.window
                p.update_cont += 1
                if p.update_cont > 0
                    for (idx, sdp) in enumerate(conic_sets.sdpcone)
                        p.target_rank[idx] = min(2 * p.target_rank[idx], sdp.sq_side)
                    end
                    p.rank_update, p.update_cont = 0, 0
                end
            end

        # Check divergence
        elseif k > p.window && comb_residual[k - p.window] < comb_residual[k] && p.rank_update > p.window
            p.update_cont += 1
            if p.update_cont > 30
                for (idx, sdp) in enumerate(conic_sets.sdpcone)
                    p.target_rank[idx] = min(2 * p.target_rank[idx], sdp.sq_side)
                end
                p.rank_update, p.update_cont = 0, 0
            end

        # Adaptive stepsizes
        elseif primal_residual[k] > 10 * opt.tol_primal && dual_residual[k] < 10 * opt.tol_dual && k > p.window
            p.beta *= (1 - p.adapt_level)
            p.primal_step /= (1 - p.adapt_level)
            if p.beta <= opt.min_beta
                p.beta = opt.min_beta
            else
                p.adapt_level *= opt.adapt_decay
            end
            if analysis
                println("Debug: Beta = $(p.beta), AdaptLevel = $(p.adapt_level)")
            end
        elseif primal_residual[k] < 10 * opt.tol_primal && dual_residual[k] > 10 * opt.tol_dual && k > p.window
            p.beta /= (1 - p.adapt_level)
            p.primal_step *= (1 - p.adapt_level)
            if p.beta >= opt.max_beta
                p.beta = opt.max_beta
            else
                p.adapt_level *= opt.adapt_decay
            end
            if analysis
                println("Debug: Beta = $(p.beta), AdaptLevel = $(p.adapt_level)")
            end
        elseif primal_residual[k] > opt.residual_relative_diff * dual_residual[k] && k > p.window
            p.beta *= (1 - p.adapt_level)
            p.primal_step /= (1 - p.adapt_level)
            if p.beta <= opt.min_beta
                p.beta = opt.min_beta
            else
                p.adapt_level *= opt.adapt_decay
            end
            if analysis
                println("Debug: Beta = $(p.beta), AdaptLevel = $(p.adapt_level)")
            end
        elseif opt.residual_relative_diff * primal_residual[k] < dual_residual[k] && k > p.window
            p.beta /= (1 - p.adapt_level)
            p.primal_step *= (1 - p.adapt_level)
            if p.beta >= opt.max_beta
                p.beta = opt.max_beta
            else
                p.adapt_level *= opt.adapt_decay
            end
            if analysis
                println("Debug: Beta = $(p.beta), AdaptLevel = $(p.adapt_level)")
            end
        end
    end

    cont = 1
    @inbounds for sdp in conic_sets.sdpcone, j in 1:sdp.sq_side, i in j:sdp.sq_side
        if i != j
            pair.x[cont] /= sqrt(2.0)
        end
        cont += 1
    end

    # Compute results
    time_ = time() - p.time0
    prim_obj = dot(c_orig, pair.x)
    dual_obj = - dot(rhs_orig, pair.y)
    slack = A_orig * pair.x - b_orig
    slack2 = G_orig * pair.x - h_orig
    res_eq = norm(slack) / (1 + norm(b_orig))
    res_dual = prim_obj - dual_obj
    gap = (prim_obj - dual_obj) / abs(prim_obj) * 100
    pair.x = pair.x[var_ordering]

    ctr_primal = Float64[]
    for soc in conic_sets.socone
        append!(ctr_primal, pair.x[soc.idx])
    end
    for sdp in conic_sets.sdpcone
        append!(ctr_primal, pair.x[sdp.vec_i])
    end

    if opt.log_verbose
        println("----------------------------------------------------------------------")
        if p.converged
            println(" Solution metrics [solved]:")
        else
            println(" Solution metrics [failed to converge]:")
        end
        println(" Primal objective = $(round(prim_obj; digits = 5))")
        println(" Dual objective = $(round(dual_obj; digits = 5))")
        println(" Duality gap (%) = $(round(gap; digits = 2)) %")
        println(" Duality residual = $(round(res_dual; digits = 5))")
        println(" ||A(X) - b|| / (1 + ||b||) = $(round(res_eq; digits = 6))")
        println(" time elapsed = $(round(time_; digits = 6))")
        println("======================================================================")
    end
    return CPResult(Int(p.converged), pair.x, pair.y, -vcat(slack, slack2, -ctr_primal), res_eq, res_dual, prim_obj, dual_obj, gap, time_)
end

function convergedrank(p::Params, cones::ConicSets, opt::Options)
    for (idx, sdp) in enumerate(cones.sdpcone)
        if !(p.min_eig[idx] < opt.tol_eig || p.target_rank[idx] > opt.max_target_rank_krylov_eigs || sdp.sq_side < opt.min_size_krylov_eigs)
            return false
        end
    end
    return true
end

function box_projection!(v::Array{Float64,1}, aff::AffineSets, step::Float64)
    # Projection onto = b
    @inbounds @simd for i in 1:length(aff.b)
        v[i] = aff.b[i]
    end
    # Projection onto <= h
    @inbounds @simd for i in 1:length(aff.h)
        v[aff.p+i] = min(v[aff.p+i] / step, aff.h[i])
    end
    return nothing
end

function compute_residual!(pair::PrimalDual, a::AuxiliaryData, primal_residual::CircularVector{Float64}, dual_residual::CircularVector{Float64}, comb_residual::CircularVector{Float64}, mat::Matrices, p::Params)
    # Compute primal residual
    a.Mty_old .+= .- a.Mty .+ (1.0 / (1.0 + p.primal_step)) .* (pair.x_old .- pair.x)
    primal_residual[p.iter] = norm(a.Mty_old, 2) / (1.0 + max(p.norm_c, maximum(abs.(a.Mty))))

    # Compute dual residual
    a.Mx_old .+= .- a.Mx .+ (1.0 / (1.0 + p.dual_step)) .* (pair.y_old .- pair.y)
    dual_residual[p.iter] = norm(a.Mx_old, 2) / (1.0 + max(p.norm_rhs, maximum(abs.(a.Mx))))

    # Compute combined residual
    comb_residual[p.iter] = primal_residual[p.iter] + dual_residual[p.iter]

    # Keep track of previous iterates
    copyto!(pair.x_old, pair.x)
    copyto!(pair.y_old, pair.y)
    copyto!(a.Mty_old, a.Mty)
    copyto!(a.Mx_old, a.Mx)
    copyto!(a.MtMx_old, a.MtMx)

    return nothing
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, affine_sets::AffineSets, mat::Matrices, opt::Options, p::Params)
    delta = .99
    cont = 0
    p.primal_step = p.primal_step * sqrt(1.0 + p.theta)
    for i in 1:opt.max_linsearch_steps
        cont += 1
        p.theta = p.primal_step / p.primal_step_old

        @timeit "linesearch 1" begin
            a.y_half .= pair.y .+ (p.beta * p.primal_step) .* ((1.0 + p.theta) .* a.Mx .- p.theta .* a.Mx_old)
        end
        @timeit "linesearch 2" begin
            # REF a.y_temp = a.y_half - beta * primal_step * box_projection(a.y_half, affine_sets, beta * primal_step)
            copyto!(a.y_temp, a.y_half)
            box_projection!(a.y_half, affine_sets, p.beta * p.primal_step)
            a.y_temp .-= (p.beta * p.primal_step) .* a.y_half
        end

        # @timeit "linesearch 3" begin
        #     a.Mty .= a.Mty_old .+ (p.beta * p.primal_step) .* ((1.0 + p.theta) .* a.MtMx .- p.theta .* a.MtMx_old)
        # end
        # @timeit "linesearch 4" if affine_sets.m == 0
        #     a.Mty .-= (p.beta * p.primal_step) .* a.Mtrhs
        # else
        #     mul!(a.Mty_aux, mat.Mt, a.y_half)
        #     a.Mty .-= (p.beta * p.primal_step) .* a.Mty_aux
        # end

        a.Mty = mat.Mt * a.y_temp
        
        # In-place norm
        @timeit "linesearch 5" begin
            a.Mty .-= a.Mty_old
            a.y_temp .-= pair.y_old
            y_norm = norm(a.y_temp)
            Mty_norm = norm(a.Mty)
        end
        if sqrt(p.beta) * p.primal_step * Mty_norm <= delta * y_norm
            break
        else
            p.primal_step *= 0.9
        end
    end

    # Reverte in-place norm
    a.Mty .+= a.Mty_old
    a.y_temp .+= pair.y_old

    copyto!(pair.y, a.y_temp)
    p.primal_step_old = p.primal_step

    return nothing
end

function preprocess!(aff::AffineSets, conic_sets::ConicSets)
    c_orig = zeros(1)
    if length(conic_sets.sdpcone) >= 1 || length(conic_sets.socone) >= 1
        all_cone_vars = Int[]
        for (idx, sdp) in enumerate(conic_sets.sdpcone)
            M = zeros(Int, sdp.sq_side, sdp.sq_side)
            iv = conic_sets.sdpcone[idx].vec_i
            im = conic_sets.sdpcone[idx].mat_i
            for i in eachindex(iv)
                M[im[i]] = iv[i]
            end
            X = Symmetric(M, :L)

            n = size(X)[1] # columns or line
            cont = 1
            sdp_vars = zeros(Int, div(sdp.sq_side*(sdp.sq_side+1), 2))
            for j in 1:n, i in j:n
                sdp_vars[cont] = X[i, j]
                cont += 1
            end
            append!(all_cone_vars, sdp_vars)
        end
        for (idx, soc) in enumerate(conic_sets.socone)
            soc_vars = copy(soc.idx)
            append!(all_cone_vars, soc_vars)
        end

        totvars = aff.n
        extra_vars = sort(collect(setdiff(Set(collect(1:totvars)),Set(all_cone_vars))))
        ord = vcat(all_cone_vars, extra_vars)
    else
        ord = collect(1:aff.n)
    end

    c_orig = copy(aff.c)

    aff.A, aff.G, aff.c = aff.A[:, ord], aff.G[:, ord], aff.c[ord]
    return c_orig[ord], sortperm(ord)
end

function map_socs!(v::Vector{Float64}, conic_sets::ConicSets, a::AuxiliaryData)
    cont = 0
    for (idx, sdp) in enumerate(conic_sets.sdpcone)
        cont += div(sdp.sq_side*(sdp.sq_side+1), 2)
    end
    sizehint!(a.soc_v, length(conic_sets.socone))
    sizehint!(a.soc_s, length(conic_sets.socone))
    for (idx, soc) in enumerate(conic_sets.socone)
        len = soc.len
        # push!(a.soc_v, view(v, cont+1:cont+len-1))
        # push!(a.soc_s, view(v, cont+len))
        push!(a.soc_s, view(v, cont+1))
        push!(a.soc_v, view(v, cont+2:cont+len))
        cont += len
    end
    return nothing
end

function primal_step!(pair::PrimalDual, a::AuxiliaryData, cones::ConicSets, mat::Matrices, arc::Vector{ARPACKAlloc{Float64}}, opt::Options, p::Params)

    pair.x .-= p.primal_step .* (a.Mty .+ mat.c)

    # Projection onto the psd cone
    if length(cones.sdpcone) >= 1
        @timeit "sdp proj" sdp_cone_projection!(pair.x, a, cones, arc, opt, p)
    end

    if length(cones.socone) >= 1
        @timeit "soc proj" so_cone_projection!(pair.x, a, cones, opt, p)
    end

    @timeit "linesearch -1" mul!(a.Mx, mat.M, pair.x)
    @timeit "linesearch 0" mul!(a.MtMx, mat.Mt, a.Mx)

    return nothing
end

function print_progress(primal_res::Float64, dual_res::Float64, p::Params)
    s_k = @sprintf("%d", p.iter)
    s_k *= " |"
    s_s = @sprintf("%.4f", primal_res + dual_res)
    s_s *= " |"
    s_p = @sprintf("%.4f", primal_res)
    s_p *= " |"
    s_d = @sprintf("%.4f", dual_res)
    s_d *= " |"
    s_target_rank = @sprintf("%.0f", sum(p.target_rank))
    s_target_rank *= " |"
    s_time = @sprintf("%.4f", time() - p.time0)
    s_time *= " |"
    a = "|"
    a *= " "^max(0, 9 - length(s_k))
    a *= s_k
    a *= " "^max(0, 12 - length(s_s))
    a *= s_s
    a *= " "^max(0, 12 - length(s_p))
    a *= s_p
    a *= " "^max(0, 12 - length(s_d))
    a *= s_d
    a *= " "^max(0, 12 - length(s_target_rank))
    a *= s_target_rank
    a *= " "^max(0, 12 - length(s_time))
    a *= s_time
    println(a)
    return nothing
end

function sdp_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, arc::Vector{ARPACKAlloc{Float64}}, opt::Options, p::Params)

    p.min_eig, current_rank, sqrt_2 = zeros(length(cones.sdpcone)), 0, sqrt(2.0)
    # Build symmetric matrix(es) X
    @timeit "reshape1" begin
        cont = 1
        @inbounds for (idx, sdp) in enumerate(cones.sdpcone), j in 1:sdp.sq_side, i in j:sdp.sq_side
            if i != j
                a.m[idx].data[i,j] = v[cont] / sqrt_2
            else
                a.m[idx].data[i,j] = v[cont]
            end
            cont += 1
        end
    end
    for (idx, sdp) in enumerate(cones.sdpcone)
        if sdp.sq_side == 1
            a.m[idx][1] = max(0.0, a.m[idx][1])
            p.min_eig[idx] = a.m[idx][1]
        elseif !opt.full_eig_decomp && p.target_rank[idx] <= opt.max_target_rank_krylov_eigs && sdp.sq_side > opt.min_size_krylov_eigs
            @timeit "eigs" begin 
                eig!(arc[idx], a.m[idx], p.target_rank[idx], p.iter)
                if hasconverged(arc[idx])
                    fill!(a.m[idx].data, 0.0)
                    for i in 1:p.target_rank[idx]
                        if unsafe_getvalues(arc[idx])[i] > 0.0
                            current_rank += 1
                            vec = unsafe_getvectors(arc[idx])[:, i]
                            LinearAlgebra.BLAS.gemm!('N', 'T', unsafe_getvalues(arc[idx])[i], vec, vec, 1.0, a.m[idx].data)
                        end
                    end
                end
            end
            if hasconverged(arc[idx])
                @timeit "get min eig" p.min_eig[idx] = minimum(unsafe_getvalues(arc[idx]))
            else
                @timeit "eigfact" full_eig!(a, idx, opt)
            end
        else
            p.min_eig[idx] = 0.0
            @timeit "eigfact" full_eig!(a, idx, opt)
        end
    end
    @timeit "reshape2" begin
        cont = 1
        @inbounds for (idx, sdp) in enumerate(cones.sdpcone), j in 1:sdp.sq_side, i in j:sdp.sq_side
            if i != j
                v[cont] = a.m[idx].data[i, j] * sqrt_2
            else
                v[cont] = a.m[idx].data[i, j]
            end
            cont += 1
        end
    end

    return nothing
end

function full_eig!(a::AuxiliaryData, idx::Int, opt::Options)
    current_rank = 0
    # fact = eigen!(a.m[1], 1e-6, Inf)
    fact = eigen!(a.m[idx])
    fill!(a.m[idx].data, 0.0)
    for i in 1:length(fact.values)
        if fact.values[i] > 0.0
            current_rank += 1
            LinearAlgebra.BLAS.gemm!('N', 'T', fact.values[i], fact.vectors[:, i], fact.vectors[:, i], 1.0, a.m[idx].data)
        end
    end
    return nothing
end

function so_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, opt::Options, p::Params)
    for (idx, soc) in enumerate(cones.socone)
        # @show "a", pair.x
        soc_projection!(a.soc_v[idx], a.soc_s[idx])
        # @show "b", pair.x
    end
    return nothing
end

function soc_projection!(v::ViewVector, s::ViewScalar)
    nv = norm(v)
    if nv <= -s[]
        s[] = 0.0
        v .= 0.0
    elseif nv <= s[]
        #do nothing
    else
        val = 0.5 * (1.0+s[]/nv)
        v .*= val
        s[] = val * nv
    end
    return nothing
end

function soc_gap(v::ViewVector, s::ViewScalar)
    return norm(v) - s[]
end

function soc_convergence(a::AuxiliaryData, cones::ConicSets, pair::PrimalDual, opt::Options, p::Params)
    for (idx, soc) in enumerate(cones.socone)
        if soc_gap(a.soc_v[idx], a.soc_s[idx]) >= opt.tol_soc
            return false
        end
    end
    return true
end
end