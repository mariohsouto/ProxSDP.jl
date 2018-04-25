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
        target_rank, rank_update, converged = 1, 0, false
        primal_residual, dual_residual, comb_residual = zeros(max_iter), zeros(max_iter), zeros(max_iter)

        # Diagonal scaling
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'
        rhs = vcat(affine_sets.b, affine_sets.h)
        S, Sinv, SM, T, Tc, TMt = diag_scaling(affine_sets, dims, M, Mt)
        mat = Matrices(SM, TMt, affine_sets.c, S, Sinv, SM, T, Tc, TMt)
        
        # Stepsize parameters and linesearch parameters
        @show primal_step = 1.0 / svds(M; nsv=1)[1][:S][1]
        dual_step = primal_step
        theta = 1.0          # Overrelaxation parameter
        adapt_level = 0.9    # Factor by which the stepsizes will be balanced 
        adapt_decay = 0.9    # Rate the adaptivity decreases over time
        l = 500              # Convergence check window
        norm_c, norm_rhs = norm(affine_sets.c), norm(rhs_orig)
    end

    update_cont = 0

    # Fixed-point loop
    @timeit "CP loop" for k in 1:max_iter

        # Primal update
        @timeit "primal" target_rank, min_eig = primal_step!(pair, a, dims, conic_sets, target_rank, mat, primal_step, arc, offdiag)::Tuple{Int64, Float64}
        # Dual update 
        @timeit "dual" dual_step!(pair, a, dims, affine_sets, mat, dual_step, theta)::Void
        # Compute residuals and update old iterates
        @timeit "logging" compute_residual!(pair, a, primal_residual, dual_residual, comb_residual, primal_step, dual_step, k, norm_c, norm_rhs)::Void
        # Print progress
        if mod(k, 1000) == 0 && opt.verbose
            print_progress(k, primal_residual[k], dual_residual[k], target_rank, time0)::Void
        end

        # Check convergence of inexact fixed-point
        rank_update += 1
        if primal_residual[k] < tol && dual_residual[k] < tol
            if min_eig < 1e-3
                converged = true
                best_prim_residual, best_dual_residual = primal_residual[k], dual_residual[k]
                print_progress(k, primal_residual[k], dual_residual[k], target_rank, time0)::Void
                break
            elseif rank_update > l
                update_cont += 1
                if update_cont > 3
                    target_rank = min(2 * target_rank, dims.n)
                    rank_update = 0
                    update_cont = 0
                end
            end

        # Check divergence
        elseif k > l && comb_residual[k - l] < comb_residual[k] && rank_update > l
            update_cont += 1
            if update_cont > 5
                rank_update, update_cont = 0, 0
                target_rank = min(2 * target_rank, dims.n)
            end

        # Adaptive stepsizes  
        # elseif primal_residual[k] > tol && dual_residual[k] < tol
        #     primal_step /= (1 - adapt_level)
        #     dual_step *= (1 - adapt_level)
        #     adapt_level *= adapt_decay
        # elseif primal_residual[k] < tol && dual_residual[k] > tol
        #     primal_step *= (1 - adapt_level)
        #     dual_step /= (1 - adapt_level)
        #     adapt_level *= adapt_decay
        # elseif primal_residual[k] > 10.0 * dual_residual[k]
        #     primal_step /= (1 - adapt_level)
        #     dual_step *= (1 - adapt_level)
        #     adapt_level *= adapt_decay
        #     rank_update = 0
        # elseif 10.0 * primal_residual[k] < dual_residual[k]
        #     primal_step *= (1 - adapt_level)
        #     dual_step /= (1 - adapt_level)
        #     adapt_level *= adapt_decay 
        end
    end

    @show prim_obj = dot(affine_sets.c, pair.x)

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
    # Base.LinAlg.axpy!(-1.0, a.Mty, a.Mty_old)
    # Base.LinAlg.axpy!((1.0 / (1.0 + primal_step)), pair.x_old, a.Mty_old)
    # Base.LinAlg.axpy!(-(1.0 / (1.0 + primal_step)), pair.x, a.Mty_old)
    # primal_residual[iter] = norm(a.Mty_old, 2) / (1.0 + norm_c)
    primal_residual[iter] = norm(pair.x - pair.x_old, 2) / (1.0 + norm_c)

    # Compute dual residual
    # Base.LinAlg.axpy!(-1.0, a.Mx, a.Mx_old)
    # Base.LinAlg.axpy!((1.0 / (1.0 + dual_step)), pair.y_old, a.Mx_old)
    # Base.LinAlg.axpy!(-(1.0 / (1.0 + dual_step)), pair.y, a.Mx_old)
    # dual_residual[iter] = norm(a.Mx_old, 2) / (1.0 + norm_rhs)
    dual_residual[iter] = norm(pair.y - pair.y_old, 2) / (1.0 + norm_rhs)

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
    div = vec(sum(abs.(M), 1) .^ 1.0)
    div[find(x-> x == 0.0, div)] = 1.0
    T = spdiagm(1.0 ./ div)
    
    # Left conditioner
    div = vec(sum(abs.(M), 2) .^ 1.0)
    div[find(x-> x == 0.0, div)] = 1.0
    S = spdiagm(1.0 ./ div)

    S, T = speye(size(S)...), speye(size(T)...) 

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

function primal_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, conic_sets::ConicSets, target_rank::Int64, mat::Matrices, primal_step::Float64, arc::ARPACKAlloc, offdiag::Set)::Tuple{Int64, Float64}
    
    # x = x - p_step * (Mty + c)
    Base.LinAlg.axpy!(-primal_step, a.Mty, pair.x)
    Base.LinAlg.axpy!(-primal_step, mat.c, pair.x)

    # Projection onto the psd cone
    target_rank, min_eig = sdp_cone_projection!(pair.x, a, dims, conic_sets, target_rank, arc, offdiag)::Tuple{Int64, Float64}

    A_mul_B!(a.Mx, mat.M, pair.x)

    return target_rank, min_eig
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

function sdp_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, dims::Dims, con::ConicSets, target_rank::Int64, arc::ARPACKAlloc, offdiag::Set)::Tuple{Int64, Float64}

    if target_rank < 1
        @show target_rank
        target_rank = 1
    end

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

    if target_rank <= max(16, dims.n/100) #&& dims.n >= 100
        @timeit "eigs" begin 
            eig!(arc, a.m, target_rank)
            if hasconverged(arc)
                fill!(a.m.data, 0.0)
                for i in 1:target_rank
                    if unsafe_getvalues(arc)[i] > 0.0
                        vec = unsafe_getvectors(arc)[:, i]
                        Base.LinAlg.BLAS.gemm!('N', 'T', unsafe_getvalues(arc)[i], vec, vec, 1.0, a.m.data)
                    end
                end
            end
        end
        if hasconverged(arc)
            @timeit "reshape2" begin
                cont = 1
                @inbounds for j in 1:n, i in j:n
                    v[cont] = a.m.data[i,j]
                    cont+=1
                end
            end
            return target_rank, minimum(unsafe_getvalues(arc))
        end
    end
    
    @timeit "eigfact" begin
        # fact = eigfact!(a.m, max(dims.n-target_rank, 1):dims.n)
        fact = eigfact!(a.m, 0.0, Inf)
        fill!(a.m.data, 0.0)
        for i in 1:length(fact[:values])
            if fact[:values][i] > 0.0
                Base.LinAlg.BLAS.gemm!('N', 'T', fact[:values][i], fact[:vectors][:, i], fact[:vectors][:, i], 1.0, a.m.data)
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

    return target_rank, 0.0
end

end