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
    u::Vector{Float64}
    u_old::Vector{Float64}
    PrimalDual(dims) = new(zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2), zeros(dims.m+dims.p), zeros(dims.m+dims.p))
end

type AuxiliaryData
    m::Symmetric{Float64,Matrix{Float64}}
    Mtu::Vector{Float64}
    Mtu_old::Vector{Float64}
    Mtu_diff::Vector{Float64}
    Mx::Vector{Float64}
    Mx_old::Vector{Float64}
    u_1::Vector{Float64}
    u_diff::Vector{Float64}
    AuxiliaryData(dims) = new(
        Symmetric(zeros(dims.n, dims.n), :L), zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2),
        zeros(dims.n*(dims.n+1)/2), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m)
    )
end

function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, dims::Dims, verbose=true, max_iter=Int(1e+5), tol=1e-4)::CPResult

    tic()
    println(" Initializing Primal-Dual Hybrid Gradient method")
    println("----------------------------------------------------------")
    println("|  iter  | comb. res | prim. res |  dual res |    rank   |")
    println("----------------------------------------------------------")

    @timeit "Init" begin
        opt = CPOptions(false, verbose)
        # Scale objective function
        c_orig, idx = preprocess!(affine_sets, dims, conic_sets)

        # Initialization
        pair = PrimalDual(dims)
        a = AuxiliaryData(dims)
        arc = ARPACKAlloc(Float64)
        target_rank = 1
        
        # logging
        rank_update = 0
        best_prim_residual, best_dual_residual = Inf, Inf
        converged, polishing = false, false
        primal_residual, dual_residual, comb_residual = zeros(max_iter), zeros(max_iter), zeros(max_iter)
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'

        # Stepsize parameters and linesearch parameters
        primal_step = sqrt(1.0 / vecnorm(M, 2))
        beta, theta = 1.0, 1.0      # Ratio (dual / primal) and overrelaxation parameter
        adapt_level = 0.5           # Factor by which the stepsizes will be balanced 
        adapt_decay = 0.99          # Rate the adaptivity decreases over time

        # Initial iterates
        pair.x[1] = 1.0
        dual_step!(pair, a, dims, affine_sets, M, beta * primal_step, theta)::Void
    end

    # Fixed-point loop
    @timeit "CP loop" for k in 1:max_iter

        # Update primal variable
        @timeit "primal" target_rank, min_eig = primal_step!(pair, a, dims, conic_sets, target_rank, Mt, affine_sets.c, primal_step, arc)::Tuple{Int64, Float64}

        # Dual update with linesearch
        @timeit "linesearch" primal_step, dual_step, beta = linesearch!(pair, a, dims, affine_sets, M, Mt, primal_step, beta, theta)::Tuple{Float64, Float64, Float64}

        # Compute residuals and update old iterates
        @timeit "logging" compute_residual(pair, a, primal_residual, dual_residual, comb_residual, primal_step, dual_step, k)::Void

        # Print progress
        if mod(k, 1000) == 0 && opt.verbose
            print_progress(k, primal_residual[k], dual_residual[k], target_rank)::Void
        end

        # Check convergence
        rank_update += 1
        if comb_residual[k] < tol
            # Check convergence of inexact fixed-point
            @timeit "primal" target_rank, min_eig = primal_step!(pair, a, dims, conic_sets, target_rank + 1, Mt, affine_sets.c, primal_step, arc)::Tuple{Int64, Float64}

            if min_eig < tol
                print_progress(k, primal_residual[k], dual_residual[k], target_rank)::Void
                converged = true
                best_prim_residual, best_dual_residual = primal_residual[k], dual_residual[k]
                break
            elseif rank_update > 1000
                target_rank *= 2
                rank_update = 0
                adapt_level = 0.5
            end

        # Check divergence
        elseif k > 2000 && comb_residual[k - 1999] < comb_residual[k] && rank_update > 1000
            target_rank *= 2
            rank_update = 0
            adapt_level = 0.5
            print_progress(k, primal_residual[k], dual_residual[k], target_rank)::Void

        # Adaptive beta
        elseif primal_residual[k] > tol && dual_residual[k] < tol 
            beta = max(1e-3, min(beta * (1 - adapt_level), 1e+3))
            adapt_level *= adapt_decay 
        elseif primal_residual[k] < tol && dual_residual[k] > tol
            beta = max(1e-3, min(beta / (1 + adapt_level), 1e+3))
            adapt_level *= adapt_decay 
        end
    end

    time = toq()
    println("Time = $time")
    @show dot(c_orig, pair.x)
    pair.x = pair.x[idx]

    return CPResult(Int(converged), pair.x, pair.u, 0.0*pair.x, best_prim_residual, best_dual_residual, dot(c_orig[idx], pair.x))
end

function compute_residual(pair::PrimalDual, a::AuxiliaryData, primal_residual::Array{Float64,1}, dual_residual::Array{Float64,1}, comb_residual::Array{Float64,1}, primal_step::Float64, dual_step::Float64, iter::Int64)::Void    
    # Compute primal residual
    Base.LinAlg.axpy!(-1.0, a.Mtu, a.Mtu_old)
    Base.LinAlg.axpy!((1.0 / primal_step), pair.x_old, a.Mtu_old)
    Base.LinAlg.axpy!(-(1.0 / primal_step), pair.x, a.Mtu_old)
    primal_residual[iter] = norm(a.Mtu_old, 2) / (1.0 + norm(pair.x_old, 2))

    # Compute dual residual
    Base.LinAlg.axpy!(-1.0, a.Mx, a.Mx_old)
    Base.LinAlg.axpy!((1.0 / dual_step), pair.u_old, a.Mx_old)
    Base.LinAlg.axpy!(-(1.0 / dual_step), pair.u, a.Mx_old)
    dual_residual[iter] = norm(a.Mx_old, 2) / (1.0 + norm(pair.u_old, 2))

    # Compute combined residual
    comb_residual[iter] = primal_residual[iter] + dual_residual[iter]

    # Keep track of previous iterates
    copy!(pair.x_old, pair.x)
    copy!(pair.u_old, pair.u)
    copy!(a.Mtu_old, a.Mtu)
    copy!(a.Mx_old, a.Mx)
    return nothing
end

function diag_scaling(affine_sets::AffineSets, alpha::Float64, dims::Dims, M::SparseMatrixCSC{Float64,Int64}, Mt::SparseMatrixCSC{Float64,Int64})
    # Diagonal scaling
    div = vec(sum(abs.(M).^(2.0-alpha), 1))
    div[find(x-> x == 0.0, div)] = 1.0
    T = spdiagm(1.0 ./ div)
    div = vec(sum(abs.(M).^alpha, 2))
    div[find(x-> x == 0.0, div)] = 1.0
    S = spdiagm(1.0 ./ div)
    Sinv = div

    # Cache matrix multiplications
    TMt = T * Mt
    Tc = T * affine_sets.c
    SM = S * M
    rhs = vcat(affine_sets.b, affine_sets.h)
    Srhs = S * rhs
    affine_sets.b = Srhs[1:dims.p]
    affine_sets.h = Srhs[dims.p+1:end]

    return affine_sets, TMt, Tc, S, SM, Sinv
end

function dual_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, affine_sets::AffineSets, M::SparseMatrixCSC{Float64,Int64}, dual_step::Float64, theta::Float64)::Void
    
    # copy!(a.u_1, theta * a.Mx_old) # theta * K * x_old
    @inbounds @simd for i in eachindex(pair.u)
        a.u_1[i] = theta * a.Mx_old[i]
    end
    A_mul_B!(a.Mx, M, pair.x) # K * x
    Base.LinAlg.axpy!(-(1.0 + theta), a.Mx, a.u_1) #  (1 + theta) * K * x
    Base.LinAlg.axpy!(-dual_step, a.u_1, pair.u) # alpha*x + y
    @inbounds @simd for i in eachindex(pair.u)
        a.u_1[i] = pair.u[i] / dual_step
    end
    @timeit "box" box_projection!(a.u_1, dims, affine_sets)
    Base.LinAlg.axpy!(-dual_step, a.u_1, pair.u) # alpha*x + y
    return nothing
end

function box_projection(v::Vector{Float64}, dims::Dims, aff::AffineSets)::Vector{Float64}
    box_projection!(v::Vector{Float64}, dims::Dims, aff::AffineSets)
    return v
end

function box_projection!(v::Vector{Float64}, dims::Dims, aff::AffineSets)::Void
    # Projection onto = b
    @inbounds @simd for i in 1:length(aff.b)
        v[i] = aff.b[i]
    end
    # Projection onto <= h
    @inbounds @simd for i in 1:length(aff.h)
        v[dims.p+i] = min(v[dims.p+i], aff.h[i])
    end
    return nothing
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, affine_sets::AffineSets, M::SparseMatrixCSC{Float64,Int64}, Mt::SparseMatrixCSC{Float64,Int64}, primal_step::Float64, beta::Float64, theta::Float64)::Tuple{Float64, Float64, Float64}
    max_iter_linesearch = 100
    delta = 0.99
    mu = 0.6
    primal_step_old = primal_step
    primal_step = primal_step * sqrt(1.0 + theta)
    # Linesearch loop
    for i = 1:max_iter_linesearch
        # Inital guess for theta
        theta = primal_step / primal_step_old
        # Update dual variable
        @timeit "dual" dual_step!(pair, a, dims, affine_sets, M, beta * primal_step, theta)
        # Check linesearch convergence
        A_mul_B!(a.Mtu, Mt, pair.u)
        copy!(a.Mtu_diff, a.Mtu)
        Base.LinAlg.axpy!(-1.0, a.Mtu_old, a.Mtu_diff)
        copy!(a.u_diff, pair.u)
        Base.LinAlg.axpy!(-1.0, pair.u_old, a.u_diff)
        if primal_step * sqrt(beta) * norm(a.Mtu_diff) <= delta * norm(a.u_diff)
            return primal_step, beta * primal_step, beta
        else
            primal_step_old = primal_step
            primal_step *= mu
        end
    end
    return primal_step, beta * primal_step, beta
end

function primal_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, conic_sets::ConicSets, target_rank::Int64, Mt::SparseMatrixCSC{Float64,Int64}, c::Vector{Float64}, primal_step::Float64, arc::ARPACKAlloc)::Tuple{Int64, Float64}
    # x=x+(-t)*(Mt*u)+(-t)*c
    Base.LinAlg.axpy!(-primal_step, a.Mtu, pair.x) # x=x+(-t)*(Mt*u)
    Base.LinAlg.axpy!(-primal_step, c, pair.x) # x=x+(-t)*c

    # Projection onto the psd cone
    return sdp_cone_projection!(pair.x, a, dims, conic_sets, target_rank, arc)::Tuple{Int64, Float64}
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
    sdp_vars = zeros(Int, div(n*(n+1),2))
    for j in 1:n, i in j:n
        sdp_vars[cont] = X[i,j]
        cont+=1
    end

    totvars = dims.n
    extra_vars = collect(setdiff(Set(collect(1:totvars)),Set(sdp_vars)))
    ord = vcat(sdp_vars, extra_vars)

    ids = vec(X)
    offdiag_ids = setdiff(Set(ids), Set(diag(X)))
    c_orig = copy(aff.c)
    for i in offdiag_ids
        aff.c[i] /= 2.0
    end  

    aff.A, aff.G, aff.c = aff.A[:, ord], aff.G[:, ord], aff.c[ord]
    return c_orig[ord], sortperm(ord)
end

function sdp_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, dims::Dims, con::ConicSets, target_rank::Int64, arc::ARPACKAlloc)::Tuple{Int64, Float64}

    n = dims.n
    iv = con.sdpcone[1][1]::Vector{Int}
    im = con.sdpcone[1][2]::Vector{Int}
    @timeit "reshape1" begin
        cont = 1
        @inbounds for j in 1:n, i in j:n
            a.m.data[i,j] = v[cont]
            cont+=1
        end
    end

    if target_rank <= 16
        @timeit "eigs" begin 
            eig!(arc, a.m, target_rank)
            if hasconverged(arc)
                fill!(a.m.data, 0.0)
                for i in 1:target_rank
                    if unsafe_getvalues(arc)[i] > 0.0
                        vec = unsafe_getvectors(arc)[:, i]
                        Base.LinAlg.BLAS.gemm!('N', 'T', unsafe_getvalues(arc)[i], vec, vec, 1.0, a.m.data)
                        # Base.LinAlg.BLAS.syr!('L', unsafe_getvalues(arc)[i], unsafe_getvectors(arc)[:, i], a.m.data)
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
        fact = eigfact!(a.m, 0.0, Inf)
        fill!(a.m.data, 0.0)
        for i in 1:length(fact[:values])
            Base.LinAlg.BLAS.gemm!('N', 'T', fact[:values][i], fact[:vectors][:, i], fact[:vectors][:, i], 1.0, a.m.data)
        end
    end

    @timeit "reshape2" begin
        cont = 1
        @inbounds for j in 1:n, i in j:n
            v[cont] = a.m.data[i,j]
            cont+=1
        end
    end
    return length(fact[:values]), 0.0
end

function print_progress(k::Int64, primal_res::Float64, dual_res::Float64, target_rank::Int64)::Void
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
    println(a)
    return nothing
end

end