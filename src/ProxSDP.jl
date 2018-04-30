module ProxSDP

using MathOptInterface, TimerOutputs

include("mathoptinterface.jl")
include("eigsolver.jl")

immutable Dims
    n::Int  # Size of primal variables
    p::Int  # Number of linear equalities
    m::Int  # Number of linear inequalities
end

type AffineSets{T}
    A::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
    G::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
    b::Vector{T}
    h::Vector{T}
    c::Vector{T}
end

type ConicSets
    sdpcone::Vector{Tuple{Vector{Int},Vector{Int}}}
end

struct CPResult
    status::Int
    primal::Vector{Float64}
    dual::Vector{Float64}
    slack::Vector{Float64}
    primal_residual::Float64
    dual_residual::Float64
    objval::Float64
end

struct CPOptions
    fullmat::Bool
    verbose::Bool
end

type PrimalDual
    x::Vector{Float64}
    x_old::Vector{Float64}
    y::Vector{Float64}
    y_old::Vector{Float64}
    y_aux::Vector{Float64}
    PrimalDual(dims) = new(zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2), zeros(dims.m+dims.p), zeros(dims.m+dims.p), zeros(dims.m+dims.p))
end

type AuxiliaryData
    m::Symmetric{Float64,Matrix{Float64}}
    Mty::Vector{Float64}
    Mty_old::Vector{Float64}
    Mty_diff::Vector{Float64}
    Mx::Vector{Float64}
    Mx_old::Vector{Float64}

    TMty::Vector{Float64}
    TMty_old::Vector{Float64}
    SMx::Vector{Float64}
    SMx_old::Vector{Float64}
    Sproj::Vector{Float64}

    y_half::Vector{Float64}
    y_diff::Vector{Float64}
    AuxiliaryData(dims) = new(
        Symmetric(zeros(dims.n, dims.n), :L), zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2),
        zeros(dims.n*(dims.n+1)/2), zeros(dims.p+dims.m), zeros(dims.p+dims.m),
        zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m),
        zeros(dims.p+dims.m), zeros(dims.p+dims.m)
    )
end

type Matrices
    M::SparseMatrixCSC{Float64,Int64}
    Mt::SparseMatrixCSC{Float64,Int64}
    c::Vector{Float64}
    S::SparseMatrixCSC{Float64,Int64}
    Sinv::Vector{Float64}
    SM::SparseMatrixCSC{Float64,Int64}
    T::SparseMatrixCSC{Float64,Int64}
    Tc::Vector{Float64}
    TMt::SparseMatrixCSC{Float64,Int64}
    Matrices(M, Mt, c, S, Sinv, SM, T, Tc, TMt) = new(M, Mt, c, S, Sinv, SM, T, Tc, TMt)
end

function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, dims::Dims, verbose=true, max_iter=Int(3 * 1e+5), tol=1e-4)::CPResult

    if verbose
        println("======================================================================")
        println("          ProxSDP : Proximal Semidefinite Programming Solver          ")
        println("                 (c) Mario Souto and Joaquim D. Garcia, 2018          ")
        println("                                                Beta version          ")
        println("----------------------------------------------------------------------")
        println(" Initializing Primal-Dual Hybrid Gradient method")
        println("----------------------------------------------------------------------")
        println("|  iter  | comb. res | prim. res |  dual res |    rank   |  time (s) |")
        println("----------------------------------------------------------------------")
    end

    time0 = time()
    tic()
    @timeit "Init" begin
        rhs_orig = vcat(affine_sets.b, affine_sets.h)
        opt = CPOptions(false, verbose)  
        # Scale objective function
        c_orig, idx, offdiag = preprocess!(affine_sets, dims, conic_sets)
        for line in 1:dims.p
            cont = 1
            @inbounds for j in 1:dims.n, i in j:dims.n
                if i != j
                    affine_sets.A[line, cont] *= (sqrt(2.0) / 2.0)
                end
                cont += 1
            end
        end
        for line in 1:dims.m
            cont = 1
            @inbounds for j in 1:dims.n, i in j:dims.n
                if i != j
                    affine_sets.G[line, cont] *= (sqrt(2.0) / 2.0)
                end
                cont += 1
            end
        end

        cont = 1
        @inbounds for j in 1:dims.n, i in j:dims.n
            if i != j
                affine_sets.c[cont] *= (sqrt(2.0) / 2.0)
            end
            cont += 1
        end

        # Initialization
        pair = PrimalDual(dims)
        a = AuxiliaryData(dims)
        arc = ARPACKAlloc(Float64)
        target_rank, rank_update, converged, current_rank = 1, 0, false, 1
        primal_residual, dual_residual, comb_residual = zeros(max_iter), zeros(max_iter), zeros(max_iter)

        # Diagonal scaling
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'
        rhs = vcat(affine_sets.b, affine_sets.h)
        S, Sinv, SM, T, Tc, TMt = diag_scaling(affine_sets, dims, M, Mt)
        mat = Matrices(SM, TMt, affine_sets.c, S, Sinv, SM, T, Tc, TMt)
        
        # Stepsize parameters and linesearch parameters
        # @show primal_step = 1.0 / svds(M; nsv=1)[1][:S][1]
        primal_step = 1.0
        dual_step = primal_step
        theta = 1.0          # Overrelaxation parameter
        adapt_level = 0.9    # Factor by which the stepsizes will be balanced 
        adapt_decay = 0.9    # Rate the adaptivity decreases over time
        l = 500              # Convergence check window
        norm_c, norm_rhs = norm(affine_sets.c), norm(rhs_orig)
    end

    update_cont = 0
    max_eig = 1.0
    beta = 1.0
    
    # Fixed-point loop
    @timeit "CP loop" for k in 1:max_iter

        # Primal update
        @timeit "primal" target_rank, current_rank, max_eig, min_eig = primal_step!(pair, a, dims, conic_sets, k, mat, primal_step, arc, offdiag, max_eig)::Tuple{Int64, Int64, Float64, Float64}
        # Dual update 
        @timeit "dual" dual_step!(pair, a, dims, affine_sets, mat, dual_step, theta)::Void
        # @timeit "linesearch" primal_step, dual_step, beta, theta = linesearch!(pair, a, dims, affine_sets, mat, primal_step, beta, theta)::Tuple{Float64, Float64, Float64, Float64}
        # Compute residuals and update old iterates
        @timeit "logging" compute_residual!(pair, a, primal_residual, dual_residual, comb_residual, primal_step, dual_step, k, norm_c, norm_rhs)::Void
        # Print progress
        if mod(k, 100) == 0 && opt.verbose
            print_progress(k, primal_residual[k], dual_residual[k], current_rank, time0)::Void
            # @show primal_step, dual_step, adapt_level
            @show min_eig, max_eig, current_rank
        end

        # Check convergence of inexact fixed-point
        rank_update += 1
        if primal_residual[k] < tol && dual_residual[k] < tol
            # if min_eig < tol
                converged = true
                best_prim_residual, best_dual_residual = primal_residual[k], dual_residual[k]
                print_progress(k, primal_residual[k], dual_residual[k], target_rank, time0)::Void
                break
            # end

        # Check divergence
        elseif k > l && comb_residual[k - l] <= comb_residual[k] && rank_update > l
            update_cont += 1
            if update_cont > 5
                rank_update, update_cont = 0, 0
                target_rank = min(2 * target_rank, dims.n)
            end

        # Adaptive stepsizes  
        elseif primal_residual[k] > tol && dual_residual[k] < tol
            primal_step /= (1 - adapt_level)
            dual_step *= (1 - adapt_level)
            beta = dual_step / primal_step
            # beta *= (1 - adapt_level)
            adapt_level *= adapt_decay
        elseif primal_residual[k] < tol && dual_residual[k] > tol
            primal_step *= (1 - adapt_level)
            dual_step /= (1 - adapt_level)
            beta = dual_step / primal_step
            # beta /= (1 - adapt_level)
            adapt_level *= adapt_decay
        end
    end

    cont = 1
    @inbounds for j in 1:dims.n, i in j:dims.n
        if i != j
            pair.x[cont] /= sqrt(2.0)
        end
        cont += 1
    end

    # Compute results
    time_ = toq()
    prim_obj = dot(c_orig, pair.x)
    dual_obj = - dot(rhs, pair.y)
    pair.x = pair.x[idx]

    if verbose
        println("----------------------------------------------------------------------")
        if converged
            println(" Status = solved")
        else
            println(" Status = ProxSDP failed to converge")
        end
        println(" Elapsed time = $(round(time_, 2))s")
        println("----------------------------------------------------------------------")
        println(" Primal objective = $(round(prim_obj, 4))")
        println(" Dual objective = $(round(dual_obj, 4))")
        println(" Duality gap = $(round(prim_obj - dual_obj, 4))")
        println("======================================================================")
    end

    return CPResult(Int(converged), pair.x, pair.y, 0.0*pair.x, 0.0, 0.0, prim_obj)
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, affine_sets::AffineSets, mat::Matrices, primal_step::Float64, beta::Float64, theta::Float64)::Tuple{Float64, Float64, Float64, Float64}
    max_iter_linesearch = 50
    delta = 1.0 - 1e-1
    mu = 0.9
    primal_step_old = primal_step
    primal_step = primal_step * sqrt(1.0 + theta)
    pair.y_aux = copy(pair.y)

    # Linesearch loop
    for i = 1:max_iter_linesearch
        # Inital guess for theta
        theta = primal_step / primal_step_old
        # Update dual variable
        dual_step = primal_step * beta
        @timeit "dual" dual_step!(pair, a, dims, affine_sets, mat, dual_step, theta)
        # Check linesearch convergence
        copy!(a.Mty_diff, a.Mty)
        Base.LinAlg.axpy!(-1.0, a.Mty_old, a.Mty_diff)
        copy!(a.y_diff, pair.y)
        Base.LinAlg.axpy!(-1.0, pair.y_old, a.y_diff)
        if primal_step * sqrt(beta) * norm(a.Mty_diff) <= delta * norm(a.y_diff)
            return primal_step, beta * primal_step, beta, theta
        else
            pair.y = copy(pair.y_aux)
            primal_step *= mu
            if primal_step < 1e-4
                break
            end
        end
    end

    println(":")

    primal_step = primal_step_old
    theta = 1.0
    dual_step = primal_step
    @timeit "dual" dual_step!(pair, a, dims, affine_sets, mat, dual_step, theta)

    return primal_step, beta * primal_step, beta, theta
end

function box_projection!(v::Array{Float64,1}, dims::Dims, aff::AffineSets, dual_step::Float64)::Void
    # Projection onto = b
    @inbounds @simd for i in 1:length(aff.b)
        v[i] = aff.b[i]
    end
    # Projection onto <= h
    @inbounds @simd for i in 1:length(aff.h)
        v[dims.p+i] = min(v[dims.p+i] / dual_step, aff.h[i])
    end
    return nothing
end

function compute_residual!(pair::PrimalDual, a::AuxiliaryData, primal_residual::Array{Float64,1}, dual_residual::Array{Float64,1}, comb_residual::Array{Float64,1}, primal_step::Float64, dual_step::Float64, iter::Int64, norm_c::Float64, norm_rhs::Float64)::Void    
    # Compute primal residual
    Base.LinAlg.axpy!(-1.0, a.Mty, a.Mty_old)
    Base.LinAlg.axpy!((1.0 / (1.0 + primal_step)), pair.x_old, a.Mty_old)
    Base.LinAlg.axpy!(-(1.0 / (1.0 + primal_step)), pair.x, a.Mty_old)
    primal_residual[iter] = norm(a.Mty_old, 2) / (1.0 + norm_c)

    # Compute dual residual
    Base.LinAlg.axpy!(-1.0, a.Mx, a.Mx_old)
    Base.LinAlg.axpy!((1.0 / (1.0 + dual_step)), pair.y_old, a.Mx_old)
    Base.LinAlg.axpy!(-(1.0 / (1.0 + dual_step)), pair.y, a.Mx_old)
    dual_residual[iter] = norm(a.Mx_old, 2) / (1.0 + norm_rhs)

    # Compute combined residual
    comb_residual[iter] = primal_residual[iter] + dual_residual[iter]

    # Keep track of previous iterates
    copy!(pair.x_old, pair.x)
    copy!(pair.y_old, pair.y)
    copy!(a.Mty_old, a.Mty)
    copy!(a.Mx_old, a.Mx)

    return nothing
end

function diag_scaling(affine_sets::AffineSets, dims::Dims, M::SparseMatrixCSC{Float64,Int64}, Mt::SparseMatrixCSC{Float64,Int64})

    # Right conditioner
    T = vec(sum(abs.(M), 1) .^ -1.0)
    T[find(x-> x == Inf, T)] = 1.0
    T = spdiagm(T)
    
    # Left conditioner
    S = vec(sum(abs.(M), 2) .^ -1.0)
    S[find(x-> x == Inf, S)] = 1.0
    S = spdiagm(S)

    # Cache matrix multiplications
    Sinv = 1.0 ./ diag(S)
    TMt = T * Mt
    Tc = T * affine_sets.c
    SM = S * M

    affine_sets.c = T * affine_sets.c

    rhs = S * vcat(affine_sets.b, affine_sets.h)
    # Projection onto = b
    @inbounds @simd for i in 1:length(affine_sets.b)
        affine_sets.b[i] = rhs[i]
    end
    # Projection onto <= h
    @inbounds @simd for i in 1:length(affine_sets.h)
        affine_sets.h[i] = rhs[dims.p+i]
    end

    return S, Sinv, SM, T, Tc, TMt
end

function dual_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, affine_sets::AffineSets, mat::Matrices, dual_step::Float64, theta::Float64)::Void

    # Compute intermediate dual variable (y_{k + 1/2})
    # pair.y = pair.y + dual_step * mat.M * (2.0 * pair.x - pair.x_old)
    @inbounds @simd for i in eachindex(pair.y)
        a.y_half[i] = theta * a.Mx_old[i]
    end
    Base.LinAlg.axpy!(-(1.0 + theta), a.Mx, a.y_half)
    Base.LinAlg.axpy!(-dual_step, a.y_half, pair.y)

    copy!(a.y_half, pair.y)
    @timeit "box" box_projection!(a.y_half, dims, affine_sets, dual_step)

    Base.LinAlg.axpy!(-dual_step, a.y_half, pair.y)

    A_mul_B!(a.Mty, mat.Mt, pair.y)

    return nothing
end

function preprocess!(aff::AffineSets, dims::Dims, conic_sets::ConicSets)
    c_orig = zeros(1)
    M = zeros(Int, dims.n, dims.n)
    iv = conic_sets.sdpcone[1][1]
    im = conic_sets.sdpcone[1][2]
    for i in eachindex(iv)
        M[im[i]] = iv[i]
    end
    X = Symmetric(M, :L)

    n = size(X)[1] # columns or line
    cont = 1
    sdp_vars = zeros(Int, div(n*(n+1), 2))
    for j in 1:n, i in j:n
        sdp_vars[cont] = X[i, j]
        cont += 1
    end

    totvars = dims.n
    extra_vars = collect(setdiff(Set(collect(1:totvars)),Set(sdp_vars)))
    ord = vcat(sdp_vars, extra_vars)

    ids = vec(X)
    offdiag_ids = setdiff(Set(ids), Set(diag(X)))
    c_orig = copy(aff.c)

    aff.A, aff.G, aff.c = aff.A[:, ord], aff.G[:, ord], aff.c[ord]
    return c_orig[ord], sortperm(ord), offdiag_ids
end

function primal_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, conic_sets::ConicSets, target_rank::Int64, mat::Matrices, primal_step::Float64, arc::ARPACKAlloc, offdiag::Set, max_eig::Float64)::Tuple{Int64, Int64, Float64, Float64}
    
    # x = x - p_step * (Mty + c)
    Base.LinAlg.axpy!(-primal_step, a.Mty, pair.x)
    Base.LinAlg.axpy!(-primal_step, mat.c, pair.x)

    # Projection onto the psd cone
    target_rank, current_rank, max_eig, min_eig = sdp_cone_projection!(pair.x, a, dims, conic_sets, target_rank, arc, offdiag, max_eig)::Tuple{Int64, Int64, Float64, Float64}

    A_mul_B!(a.Mx, mat.M, pair.x)

    return target_rank, current_rank, max_eig, min_eig
end

function print_progress(k::Int64, primal_res::Float64, dual_res::Float64, target_rank::Int64, time0::Float64)::Void
    s_k = @sprintf("%d", k)
    s_k *= " |"
    s_s = @sprintf("%.4f", primal_res + dual_res)
    s_s *= " |"
    s_p = @sprintf("%.4f", primal_res)
    s_p *= " |"
    s_d = @sprintf("%.4f", dual_res)
    s_d *= " |"
    s_target_rank = @sprintf("%.0f", target_rank)
    s_target_rank *= " |"
    s_time = @sprintf("%.4f", time() - time0)
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

function sdp_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, dims::Dims, con::ConicSets, k::Int64, arc::ARPACKAlloc, offdiag::Set, max_eig::Float64)::Tuple{Int64, Int64, Float64, Float64}
    
    eig_tol = 1e-6
    n = dims.n
    @timeit "reshape1" begin
        cont = 1
        @inbounds for j in 1:n, i in j:n
            if i != j
                a.m.data[i,j] = v[cont] / sqrt(2.0)
            else
                a.m.data[i,j] = v[cont]
            end
            cont += 1
        end
    end

    if k < 100
        eig!(arc, a.m, 1)
        if hasconverged(arc)
            fill!(a.m.data, 0.0)
            current_rank = 1
            if unsafe_getvalues(arc)[1] > 0.0
                vec = unsafe_getvectors(arc)[:, 1]
                Base.LinAlg.BLAS.gemm!('N', 'T', unsafe_getvalues(arc)[1], vec, vec, 1.0, a.m.data)
            end
        end
    else
        @timeit "eigfact" begin
            fact = eigfact!(a.m, max_eig * exp(-k / (0.5 * dims.n)), Inf)
            fill!(a.m.data, 0.0)
            current_rank = 0
            for i in 1:length(fact[:values])
                if fact[:values][i] > 0.0
                    current_rank += 1
                    Base.LinAlg.BLAS.gemm!('N', 'T', fact[:values][i], fact[:vectors][:, i], fact[:vectors][:, i], 1.0, a.m.data)
                end
            end
        end
    end

    @timeit "reshape2" begin
        cont = 1
        @inbounds for j in 1:n, i in j:n
            if i != j
                v[cont] = a.m.data[i, j] * sqrt(2.0)
            else
                v[cont] = a.m.data[i, j]
            end
            cont+=1
        end
    end

    if k < 100
        max_eig = unsafe_getvalues(arc)[1]
        min_eig = max_eig
    elseif fact[:values]==Float64[]
        max_eig = 0.0
        min_eig = 0.0
    else
        max_eig = max(fact[:values]...)
        min_eig = min(fact[:values]...)
    end

    return current_rank, current_rank, max_eig, min_eig
end

end