
module ProxSDP

using MathOptInterface, TimerOutputs
using Compat

include("MOIWrapper.jl")
include("eigsolver.jl")


MOIU.@model _ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)

Solver(;args...) = MOIU.CachingOptimizer(_ProxSDPModelData{Float64}(), ProxSDP.Optimizer(args))

function get_solution(opt::MOIU.CachingOptimizer{Optimizer,T}) where T
    return opt.optimizer.sol
end

# --------------------------------
mutable struct Options
    log_verbose::Bool
    timer_verbose::Bool
    max_iter::Int
    tol_primal::Float64
    tol_dual::Float64
    tol_eig::Float64

    initial_theta::Float64
    initial_beta::Float64
    min_beta::Float64
    max_beta::Float64
    initial_adapt_level::Float64
    adapt_decay::Float64 # Rate the adaptivity decreases over time
    convergence_window::Int

    residual_relative_diff::Float64

    max_linsearch_steps::Int

    max_target_rank_krylov_eigs::Int
    min_size_krylov_eigs::Int

    function Options()
        opt = new()
        opt.log_verbose = false
        opt.timer_verbose = false
        opt.max_iter = Int(1e+5)
        opt.tol_primal = 1e-3
        opt.tol_dual = 1e-3
        opt.tol_eig = 1e-3

        opt.initial_theta = 1.0
        opt.initial_beta = 1.0
        opt.min_beta = 1e-3
        opt.max_beta = 1e+3
        opt.initial_adapt_level = 0.9
        opt.adapt_decay = 0.9
        opt.convergence_window = 100

        opt.residual_relative_diff = 100.0

        opt.max_linsearch_steps = 1000

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

type AffineSets
    n::Int  # Size of primal variables
    p::Int  # Number of linear equalities
    m::Int  # Number of linear inequalities
    A::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
    G::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
    b::Vector{Float64}
    h::Vector{Float64}
    c::Vector{Float64}
end

type SDPSet
    vec_i::Vector{Int}
    mat_i::Vector{Int}
    tri_len::Int
    sq_len::Int
    sq_side::Int
end

type ConicSets
    sdpcone::Vector{SDPSet}
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

type PrimalDual
    x::Vector{Float64}
    x_old::Vector{Float64}

    y::Vector{Float64}
    y_old::Vector{Float64}
    y_aux::Vector{Float64}

    PrimalDual(aff) = new(
        zeros(aff.n), zeros(aff.n), zeros(aff.m+aff.p), zeros(aff.m+aff.p), zeros(aff.m+aff.p)
    )
end

type AuxiliaryData
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
    function AuxiliaryData(aff::AffineSets, cones::ConicSets) 
        new([Symmetric(zeros(sdp.sq_side, sdp.sq_side), :L) for sdp in cones.sdpcone], zeros(aff.n), zeros(aff.n),
        zeros(aff.n), zeros(aff.p+aff.m), zeros(aff.p+aff.m), zeros(aff.p+aff.m), 
        zeros(aff.p+aff.m), zeros(aff.n), zeros(aff.n), zeros(aff.n)
    )
    end
end

type Matrices
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

    p.theta = opt.initial_theta           # Overrelaxation parameter
    p.adapt_level = opt.initial_adapt_level     # Factor by which the stepsizes will be balanced 
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

    tic()
    @timeit "Init" begin
        # Scale objective function
        c_orig, var_ordering = preprocess!(affine_sets, conic_sets)
        A_orig, b_orig = copy(affine_sets.A), copy(affine_sets.b)
        G_orig, h_orig = copy(affine_sets.G), copy(affine_sets.h)
        rhs_orig = vcat(affine_sets.b, affine_sets.h)
        @timeit "Norm Scaling" norm_scaling(affine_sets, conic_sets)
        # Initialization
        pair = PrimalDual(affine_sets)
        a = AuxiliaryData(affine_sets, conic_sets)
        arc = [ARPACKAlloc(Float64, 1) for i in conic_sets.sdpcone]

        primal_residual, dual_residual, comb_residual = zeros(opt.max_iter), zeros(opt.max_iter), zeros(opt.max_iter)

        # Diagonal scaling
        # @show affine_sets.A, affine_sets.G
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'
        rhs = vcat(affine_sets.b, affine_sets.h)
        mat = Matrices(M, Mt, affine_sets.c)
        # @show a.Mtrhs, mat.Mt, rhs
        A_mul_B!(a.Mtrhs, mat.Mt, rhs)
        
        # Stepsize parameters and linesearch parameters
        # primal_step = sqrt(min(aff.n^2, aff.m + aff.p)) / vecnorm(M)
        p.primal_step = 1.0 / svds(M; nsv=1)[1][:S][1]
        # dual_step = primal_step
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
        if mod(k, p.window) == 0 && opt.log_verbose
            print_progress(primal_residual[k], dual_residual[k], p)
        end

        # Check convergence of inexact fixed-point
        p.rank_update += 1
        if primal_residual[k] < opt.tol_primal && dual_residual[k] < opt.tol_dual && k > 50
            if convergedrank(p, conic_sets, opt)
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
        elseif k > p.window && comb_residual[k - p.window] < 0.8 * comb_residual[k] && p.rank_update > p.window
            p.update_cont += 1
            if p.update_cont > 30
                for (idx, sdp) in enumerate(conic_sets.sdpcone)
                    p.target_rank[idx] = min(2 * p.target_rank[idx], sdp.sq_side)
                end
                p.rank_update, p.update_cont = 0, 0
            end

        # Adaptive stepsizes
        elseif primal_residual[k] > opt.tol_primal && dual_residual[k] < opt.tol_dual && k > p.window
            p.beta *= (1 - p.adapt_level)
            if p.beta <= opt.min_beta
                p.beta = opt.min_beta
            else
                p.adapt_level *= opt.adapt_decay
            end
            if analysis
                println("Debug: Beta = $(p.beta), AdaptLevel = $(p.adapt_level)")
            end
        elseif primal_residual[k] < opt.tol_primal && dual_residual[k] > opt.tol_dual && k > p.window
            p.beta /= (1 - p.adapt_level)
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

    if opt.log_verbose
        println("----------------------------------------------------------------------")
        if p.converged
            println(" Solution metrics [solved]:")
        else
            println(" Solution metrics [failed to converge]:")
        end
        println(" Primal objective = $(round(prim_obj, 5))")
        println(" Dual objective = $(round(dual_obj, 5))")
        println(" Duality gap (%) = $(round(gap, 2)) %")
        println(" Duality residual = $(round(res_dual, 5))")
        println(" ||A(X) - b|| / (1 + ||b||) = $(round(res_eq, 6))")
        println(" time elapsed = $(round(time_, 6))")
        println("======================================================================")
    end
    return CPResult(Int(p.converged), pair.x, pair.y, -vcat(slack, slack2), res_eq, res_dual, prim_obj, dual_obj, gap, time_)
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

function compute_residual!(pair::PrimalDual, a::AuxiliaryData, primal_residual::Array{Float64,1}, dual_residual::Array{Float64,1}, comb_residual::Array{Float64,1}, mat::Matrices, p::Params)
    # Compute primal residual
    a.Mty_old .+= .- a.Mty .+ (1.0 / (1.0 + p.primal_step)).*(pair.x_old .- pair.x)

    primal_residual[p.iter] = norm(a.Mty_old, 2) / (1.0 + p.norm_c)

    # Compute dual residual
    a.Mx_old .+= .- a.Mx .+ (1.0 / (1.0 + p.dual_step)) .* (pair.y_old .- pair.y)

    dual_residual[p.iter] = norm(a.Mx_old, 2) / (1.0 + p.norm_rhs)

    # Compute combined residual
    comb_residual[p.iter] = primal_residual[p.iter] + dual_residual[p.iter]

    # Keep track of previous iterates
    copy!(pair.x_old, pair.x)
    copy!(pair.y_old, pair.y)
    copy!(a.Mty_old, a.Mty)
    copy!(a.Mx_old, a.Mx)
    copy!(a.MtMx_old, a.MtMx)

    return nothing
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, affine_sets::AffineSets, mat::Matrices, opt::Options, p::Params)
    # theta = 1.0
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
            copy!(a.y_temp, a.y_half)
            box_projection!(a.y_half, affine_sets, p.beta * p.primal_step)
            a.y_temp .-= (p.beta * p.primal_step) .* a.y_half
        end

        @timeit "linesearch 3" begin
            a.Mty .= a.Mty_old .+ (p.beta * p.primal_step) .* ((1.0 + p.theta) .* a.MtMx .- p.theta .* a.MtMx_old)
        end
        @timeit "linesearch 4" if affine_sets.m == 0
            a.Mty .-= (p.beta * p.primal_step) .* a.Mtrhs
        else
            A_mul_B!(a.Mty_aux, mat.Mt, a.y_half)
            a.Mty .-= (p.beta * p.primal_step) .* a.Mty_aux
        end
        
        # In-place norm
        @timeit "linesearch 5" begin
            a.Mty .-= a.Mty_old
            a.y_temp .-= pair.y_old
            y_norm = norm(a.y_temp)
            Mty_norm = norm(a.Mty)
        end
        if sqrt(p.beta) * p.primal_step * Mty_norm <= (1.0 - 1e-3) * y_norm
            break
        else
            p.primal_step *= 0.9
        end
    end

    # Reverte in-place norm
    a.Mty .+= a.Mty_old
    a.y_temp .+= pair.y_old

    copy!(pair.y, a.y_temp)
    p.primal_step_old = p.primal_step

    return nothing
end

function dual_step!(pair::PrimalDual, a::AuxiliaryData, affine_sets::AffineSets, mat::Matrices, p::Params)
    # Compute intermediate dual variable (y_{k + 1/2})
    # pair.y = pair.y + dual_step * mat.M * (2.0 * pair.x - pair.x_old)
    a.y_half .= p.theta .* a.Mx_old .- (1.0 + p.theta) .* a.Mx
    pair.y .-= p.dual_step .* a.y_half

    copy!(a.y_half, pair.y)
    @timeit "box" box_projection!(a.y_half, affine_sets, p.dual_step)
    pair.y .-= p.dual_step .* a.y_half

    A_mul_B!(a.Mty, mat.Mt, pair.y)

    return nothing
end

function preprocess!(aff::AffineSets, conic_sets::ConicSets)
    c_orig = zeros(1)
    if length(conic_sets.sdpcone) >= 1
        all_sdp_vars = Int[]
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
            append!(all_sdp_vars,sdp_vars)
        end

        totvars = aff.n
        extra_vars = sort(collect(setdiff(Set(collect(1:totvars)),Set(all_sdp_vars))))
        ord = vcat(all_sdp_vars, extra_vars)
    else
        ord = collect(1:aff.n)
    end

    c_orig = copy(aff.c)

    aff.A, aff.G, aff.c = aff.A[:, ord], aff.G[:, ord], aff.c[ord]
    return c_orig[ord], sortperm(ord)
end

function primal_step!(pair::PrimalDual, a::AuxiliaryData, cones::ConicSets, mat::Matrices, arc::Vector{ARPACKAlloc{Float64}}, opt::Options, p::Params)

    # x = x - p_step * (Mty + c)
    pair.x .-= p.primal_step .* (a.Mty .+ mat.c)

    # Projection onto the psd cone
    if length(cones.sdpcone) >= 1
        @timeit "sdp proj" sdp_cone_projection!(pair.x, a, cones, arc, pair, opt, p)
    end

    @timeit "linesearch -1" A_mul_B!(a.Mx, mat.M, pair.x)
    @timeit "linesearch 0" A_mul_B!(a.MtMx, mat.Mt, a.Mx)

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

function sdp_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, arc::Vector{ARPACKAlloc{Float64}}, pair::PrimalDual, opt::Options, p::Params)

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
        elseif p.target_rank[idx] <= opt.max_target_rank_krylov_eigs && sdp.sq_side > opt.min_size_krylov_eigs
            @timeit "eigs" begin 
                eig!(arc[idx], a.m[idx], p.target_rank[idx], p.iter)
                if hasconverged(arc[idx])
                    fill!(a.m[idx].data, 0.0)
                    for i in 1:p.target_rank[idx]
                        if unsafe_getvalues(arc[idx])[i] > 0.0
                            current_rank += 1
                            vec = unsafe_getvectors(arc[idx])[:, i]
                            Base.LinAlg.BLAS.gemm!('N', 'T', unsafe_getvalues(arc[idx])[i], vec, vec, 1.0, a.m[idx].data)
                        end
                    end
                end
            end
            if hasconverged(arc[idx])
                @timeit "get min eig" p.min_eig[idx] = minimum(unsafe_getvalues(arc[idx]))
            end    
        else
            p.min_eig[idx] = 0.0
            @timeit "eigfact" begin
                current_rank = 0
                # fact = eigfact!(a.m[1], 1e-6, Inf)
                fact = eigfact!(a.m[idx])
                fill!(a.m[idx].data, 0.0)
                for i in 1:length(fact[:values])
                    if fact[:values][i] > 0.0
                        current_rank += 1
                        Base.LinAlg.BLAS.gemm!('N', 'T', fact[:values][i], fact[:vectors][:, i], fact[:vectors][:, i], 1.0, a.m[idx].data)
                    end
                end
            end
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
end