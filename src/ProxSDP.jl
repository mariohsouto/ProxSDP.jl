
module ProxSDP

using MathOptInterface, TimerOutputs
using Compat

include("MOIWrapper.jl")
include("eigsolver.jl")

immutable Dims
    n::Int  # Size of primal variables
    p::Int  # Number of linear equalities
    m::Int  # Number of linear inequalities
    s::Vector{Int} # Side of square matrices
end

type AffineSets{T}
    A::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
    G::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
    b::Vector{T}
    h::Vector{T}
    c::Vector{T}
end

type SDPSet
    vec_i::Vector{Int}
    mat_i::Vector{Int}
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

    PrimalDual(dims) = new(
        zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2), zeros(dims.m+dims.p), zeros(dims.m+dims.p), zeros(dims.m+dims.p)
    )
end

type AuxiliaryData
    m::Vector{Symmetric{Float64,Matrix{Float64}}}
    Mty::Vector{Float64}
    Mty_old::Vector{Float64}
    Mx::Vector{Float64}
    Mx_old::Vector{Float64}
    y_half::Vector{Float64}
    y_temp::Vector{Float64}
    MtMx::Vector{Float64}
    MtMx_old::Vector{Float64}
    Mtrhs::Vector{Float64}
    function AuxiliaryData(dims) 
        new([Symmetric(zeros(i, i), :L) for i in dims.s], zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2),
        zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m),
        zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2)
    )
    end
end

type Matrices
    M::SparseMatrixCSC{Float64,Int64}
    Mt::SparseMatrixCSC{Float64,Int64}
    c::Vector{Float64}
    Matrices(M, Mt, c) = new(M, Mt, c)
end

function printheader()
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

function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, dims::Dims, verbose=true, max_iter=Int(1e+5), tol=1e-4)::CPResult

    if verbose
        printheader()
    end

    time0 = time()
    tic()
    @timeit "Init" begin
    
    opt = CPOptions(false, verbose)  
    # Scale objective function
    c_orig, idx, offdiag = preprocess!(affine_sets, dims, conic_sets)
    A_orig, b_orig = copy(affine_sets.A), copy(affine_sets.b)
    rhs_orig = vcat(affine_sets.b, affine_sets.h)
    
    cont = 1
    
    @timeit "Scaling" begin
        cte = (sqrt(2.0) / 2.0)
        rows = rowvals(affine_sets.A)
        m, n = size(affine_sets.A)
        cont = 1
        for j in 1:dims.n, i in j:dims.n
            if i != j
                for line in nzrange(affine_sets.A, cont)
                    affine_sets.A[rows[line], cont] *= cte
                end
            end
            cont += 1
        end
        rows = rowvals(affine_sets.G)
        m, n = size(affine_sets.G)
        cont = 1
        for j in 1:dims.n, i in j:dims.n
            if i != j
                for line in nzrange(affine_sets.G, cont)
                    affine_sets.G[rows[line], cont] *= cte
                end
            end
            cont += 1
        end
        # @inbounds for j in 1:dims.n, i in j:dims.n
        #     if i != j
        #         for line in 1:dims.p
        #             affine_sets.A[line, cont] *= cte
        #         end
        #     end
        #     cont += 1
        # end
        # cont = 1
        # @inbounds for j in 1:dims.n, i in j:dims.n
        #     if i != j
        #         for line in 1:dims.m
        #             affine_sets.G[line, cont] *= cte
        #         end
        #     end
        #     cont += 1
        # end
        cont = 1
        @inbounds for j in 1:dims.n, i in j:dims.n
            if i != j
                affine_sets.c[cont] *= cte
            end
            cont += 1
        end
    end

        # Initialization
        pair = PrimalDual(dims)
        a = AuxiliaryData(dims)
        arc = ARPACKAlloc(Float64, 1)
        target_rank, rank_update, converged, update_cont = 1, 0, false, 0
        primal_residual, dual_residual, comb_residual = zeros(max_iter), zeros(max_iter), zeros(max_iter)

        # Diagonal scaling
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'
        rhs = vcat(affine_sets.b, affine_sets.h)
        mat = Matrices(M, Mt, affine_sets.c)
        A_mul_B!(a.Mtrhs, mat.Mt, rhs)
        
        # Stepsize parameters and linesearch parameters
        # primal_step = max(1.0, sqrt(min(dims.n*(dims.n+1)/2, dims.m + dims.p)) / vecnorm(M))
        primal_step = sqrt(min(dims.n*(dims.n+1)/2, dims.m + dims.p)) / vecnorm(M)
        primal_step_old = primal_step
        dual_step = primal_step
        theta = 1.0           # Overrelaxation parameter
        adapt_level = 0.9     # Factor by which the stepsizes will be balanced 
        adapt_decay = 0.9     # Rate the adaptivity decreases over time
        l = 100               # Convergence check window
        norm_c, norm_rhs = norm(affine_sets.c), norm(rhs)

        pair.x[1] = 1.0
        beta = 1.0
    end

    # Fixed-point loop
    tic()
    @timeit "CP loop" for k in 1:max_iter

        # Primal update
        @timeit "primal" current_rank, min_eig = primal_step!(pair, a, dims, target_rank, mat, primal_step, arc, offdiag, k)::Tuple{Int64, Float64}
        # Linesearch
        primal_step, primal_step_old = linesearch!(pair, a, dims, affine_sets, mat, primal_step, primal_step_old, beta)::Tuple{Float64, Float64}
        # Compute residuals and update old iterates
        compute_residual!(pair, a, primal_residual, dual_residual, comb_residual, primal_step, dual_step, k, norm_c, norm_rhs, mat)::Void
        # Print progress
        if mod(k, 100) == 0 && opt.verbose
            print_progress(k, primal_residual[k], dual_residual[k], current_rank, time0)::Void
        end

        # Check convergence of inexact fixed-point
        rank_update += 1
        if primal_residual[k] < tol && dual_residual[k] < tol
            if min_eig < tol
                converged = true
                best_prim_residual, best_dual_residual = primal_residual[k], dual_residual[k]
                print_progress(k, primal_residual[k], dual_residual[k], target_rank, time0)::Void
                break
            elseif rank_update > l
                update_cont += 1
                if update_cont > 0
                    target_rank = min(2 * target_rank, dims.n)
                    rank_update, update_cont = 0, 0
                end
            end

        # Check divergence
        elseif k > l && comb_residual[k - l] < 0.8 * comb_residual[k] && rank_update > l
            update_cont += 1
            if update_cont > 10
                target_rank = min(2 * target_rank, dims.n)
                rank_update, update_cont = 0, 0
            end

        # Adaptive stepsizes  
        elseif primal_residual[k] > tol && dual_residual[k] < tol
            beta *= (1 - adapt_level)
            adapt_level *= adapt_decay
        elseif primal_residual[k] < tol && dual_residual[k] > tol
            beta /= (1 - adapt_level)
            adapt_level *= adapt_decay
        end
    end

    toc()
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
    dual_obj = - dot(rhs_orig, pair.y)
    res_eq = norm(A_orig * pair.x - b_orig) / (1 + norm(b_orig))

    pair.x = pair.x[idx]

    if verbose
        println("----------------------------------------------------------------------")
        if converged
            println(" Solution metrics [solved]:")
        else
            println(" Solution metrics [failed to converge]:")
        end
        println(" Primal objective = $(round(prim_obj, 6))")
        println(" Dual objective = $(round(dual_obj, 6))")
        println(" Duality gap = $(round(prim_obj - dual_obj, 6))")
        println(" ||A(X) - b|| / (1 + ||b||) = $(round(res_eq, 6))")
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

function compute_residual!(pair::PrimalDual, a::AuxiliaryData, primal_residual::Array{Float64,1}, dual_residual::Array{Float64,1}, comb_residual::Array{Float64,1}, primal_step::Float64, dual_step::Float64, iter::Int64, norm_c::Float64, norm_rhs::Float64, mat::Matrices)::Void    
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
    copy!(a.MtMx_old, a.MtMx)

    return nothing
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, affine_sets::AffineSets, mat::Matrices, primal_step::Float64, primal_step_old::Float64, beta::Float64)::Tuple{Float64, Float64}
    theta = 1.0
    cont = 0
    for i in 1:100
        cont += 1
        theta = primal_step / primal_step_old

        # a.y_half = pair.y + beta * primal_step * ((1 + theta) * a.Mx - theta * a.Mx_old)
        @timeit "linesearch 1" copy!(a.y_half, pair.y)
        @timeit "linesearch 2" Base.LinAlg.axpy!(beta * primal_step * (1 + theta), a.Mx, a.y_half)
        @timeit "linesearch 3" Base.LinAlg.axpy!(-beta * primal_step * theta, a.Mx_old, a.y_half)

        # a.y_temp = a.y_half - beta * primal_step * box_projection(a.y_half, dims, affine_sets, beta * primal_step)
        @timeit "linesearch 4" copy!(a.y_temp, a.y_half)
        @timeit "linesearch 5" box_projection!(a.y_half, dims, affine_sets, beta * primal_step)
        @timeit "linesearch 6" Base.LinAlg.axpy!(-beta * primal_step, a.y_half, a.y_temp)

        # Compute Mt * y for the linesearch stopping criteria
        # @timeit "linesearch 9" A_mul_B!(a.Mty, mat.Mt, a.y_temp)
        @timeit "linesearch 7" copy!(a.Mty, a.Mty_old)
        @timeit "linesearch 8" Base.LinAlg.axpy!(beta * primal_step * (1 + theta), a.MtMx, a.Mty)
        @timeit "linesearch 9" Base.LinAlg.axpy!(-beta * primal_step * theta, a.MtMx_old, a.Mty)
        @timeit "linesearch 10" Base.LinAlg.axpy!(-beta * primal_step, a.Mtrhs, a.Mty)
        
        # In-place norm
        @timeit "linesearch 11" Base.LinAlg.axpy!(-1.0, a.Mty_old, a.Mty)
        @timeit "linesearch 12" Base.LinAlg.axpy!(-1.0, pair.y_old, a.y_temp)
        y_norm = norm(a.y_temp)
        Mty_norm = norm(a.Mty)
        if sqrt(beta) * primal_step * Mty_norm <= y_norm #(1 - 1e-16) * y_norm
            break
        else
            primal_step *= 0.9
        end
    end

    # @show cont

    # Reverte in-place norm
    Base.LinAlg.axpy!(1.0, a.Mty_old, a.Mty)
    Base.LinAlg.axpy!(1.0, pair.y_old, a.y_temp)

    copy!(pair.y, a.y_temp)
    # Update primal step
    primal_step_old = primal_step
    primal_step = primal_step * sqrt(1 + theta)

    return primal_step, primal_step_old
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
    if length(conic_sets.sdpcone) >= 1
        iv = conic_sets.sdpcone[1].vec_i
        im = conic_sets.sdpcone[1].mat_i
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
    else
        ord = collect(1:dims.n)
        offdiag_ids = Set{Int}()
    end

    c_orig = copy(aff.c)

    aff.A, aff.G, aff.c = aff.A[:, ord], aff.G[:, ord], aff.c[ord]
    return c_orig[ord], sortperm(ord), offdiag_ids
end

function primal_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, target_rank::Int64, mat::Matrices, primal_step::Float64, arc::ARPACKAlloc, offdiag::Set{Int}, iter::Int64)::Tuple{Int64, Float64}

    # x = x - p_step * (Mty + c)
    Base.LinAlg.axpy!(-primal_step, a.Mty, pair.x)
    Base.LinAlg.axpy!(-primal_step, mat.c, pair.x)

    # Projection onto the psd cone
    if length(dims.s) == 1
        @timeit "sdp proj" current_rank, min_eig = sdp_cone_projection!(pair.x, a, dims, target_rank, arc, offdiag, iter)::Tuple{Int64, Float64}
    end

    @timeit "linesearch -1" A_mul_B!(a.Mx, mat.M, pair.x)
    @timeit "linesearch 0" A_mul_B!(a.MtMx, mat.Mt, a.Mx)
    
    return current_rank, min_eig
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

function sdp_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, dims::Dims, target_rank::Int64, arc::ARPACKAlloc, offdiag::Set, iter::Int64)::Tuple{Int64, Float64}

    min_eig, current_rank, sqrt_2 = 0.0, 0, sqrt(2.0)
    # Build symmetric matrix X
    n = dims.n
    @timeit "reshape1" begin
        cont = 1
        @inbounds for j in 1:n, i in j:n
            if i != j
                a.m[1].data[i,j] = v[cont] / sqrt_2
            else
                a.m[1].data[i,j] = v[cont]
            end
            cont += 1
        end
    end

    if target_rank <= 16
        @timeit "eigs" begin 
            eig!(arc, a.m[1], target_rank, iter)
            if hasconverged(arc)
                fill!(a.m[1].data, 0.0)
                for i in 1:target_rank
                    if unsafe_getvalues(arc)[i] > 1e-10
                        current_rank += 1
                        vec = unsafe_getvectors(arc)[:, i]
                        Base.LinAlg.BLAS.gemm!('N', 'T', unsafe_getvalues(arc)[i], vec, vec, 1.0, a.m[1].data)
                    end
                end
            end
        end
    end

    if target_rank <= 16 && hasconverged(arc)
        @timeit "get min eig" min_eig = minimum(unsafe_getvalues(arc))
    else
        min_eig = 0.0
        @timeit "eigfact" begin
            current_rank = 0
            fact = eigfact!(a.m[1], 1e-10, Inf)
            # fact = eigfact!(a.m[1])
            fill!(a.m[1].data, 0.0)
            for i in 1:length(fact[:values])
                if fact[:values][i] > 0.0
                    current_rank += 1
                    Base.LinAlg.BLAS.gemm!('N', 'T', fact[:values][i], fact[:vectors][:, i], fact[:vectors][:, i], 1.0, a.m[1].data)
                end
            end
        end
    end

    cont = 1
    @timeit "reshape2" begin
        @inbounds for j in 1:n, i in j:n
            if i != j
                v[cont] = a.m[1].data[i, j] * sqrt_2
            else
                v[cont] = a.m[1].data[i, j]
            end
            cont += 1
        end
    end

    return max(current_rank, 1), min_eig
end
end