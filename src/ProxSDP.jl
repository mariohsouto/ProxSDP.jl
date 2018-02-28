module ProxSDP

using MathOptInterface, TimerOutputs

include("mathoptinterface.jl")

immutable Dims
    n::Int  # Size of primal variables
    p::Int  # Number of linear equalities
    m::Int  # Number of linear inequalities
end

type AffineSets
    A::AbstractMatrix
    G::AbstractMatrix
    b::AbstractVector
    h::AbstractVector
    c::AbstractVector
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

type AllocatedData
    m::Symmetric{Float64,Matrix{Float64}}#Matrix{Float64}
    x_1::Vector{Float64}
    u_1::Vector{Float64}
    u_2::Vector{Float64}
    AllocatedData(dims) = new(Symmetric(zeros(dims.n, dims.n), :L), zeros(dims.n*(dims.n+1)/2), zeros(dims.p+dims.m), zeros(dims.p+dims.m))
end

function chambolle_pock(affine_sets, conic_sets, dims, verbose=true, max_iter=Int(1e+5), primal_tol=1e-4, dual_tol=1e-3)   

    @timeit "Init" begin
        opt = CPOptions(false, verbose)
        c_orig = zeros(1)
        # use algorithm in full square matrix mode or triangular mode
        if opt.fullmat
            M = zeros(Int, dims.n, dims.n)
            iv = conic_sets.sdpcone[1][1]
            im = conic_sets.sdpcone[1][2]
            for i in eachindex(iv)
                M[im[i]] = iv[i]
            end
            X = Symmetric(M,:L)
            ids = vec(X)

            offdiag_ids = setdiff(Set(ids),Set(diag(X)))
            for i in offdiag_ids
                affine_sets.c[i] /= 2.0
            end
            
            # modify A, G and c
            affine_sets.A = affine_sets.A[:,ids]
            affine_sets.G = affine_sets.G[:,ids]
            affine_sets.c = affine_sets.c[ids]
            c_orig = copy(affine_sets.c)
            
        else  
            M = zeros(Int, dims.n, dims.n)
            iv = conic_sets.sdpcone[1][1]
            im = conic_sets.sdpcone[1][2]
            for i in eachindex(iv)
                M[im[i]] = iv[i]
            end
            X = Symmetric(M,:L)
            ids = vec(X)

            offdiag_ids = setdiff(Set(ids), Set(diag(X)))
            c_orig = copy(affine_sets.c)
            for i in offdiag_ids
                affine_sets.c[i] /= 2.0
            end  
        end
    end

    converged = false
    tic()
    println(" Initializing Primal-Dual Hybrid Gradient method")
    println("----------------------------------------------------------")
    println("|  iter  | comb. res | prim. res |  dual res |    rho    |")
    println("----------------------------------------------------------")


    @timeit "Init" begin
    # Given
    m, n, p = dims.m, dims.n, dims.p
    theta = 1.0 # Overrelaxation parameter

    # logging
    best_comb_residual, comb_res_rank_update = Inf, Inf
    converged, status, polishing = false, false, false
    comb_residual, dual_residual, primal_residual = Float64[], Float64[], Float64[]
    sizehint!(comb_residual, max_iter)
    sizehint!(dual_residual, max_iter)
    sizehint!(primal_residual, max_iter)

    # Primal variables
    x, x_old =  if opt.fullmat
        zeros(n^2), zeros(n^2)
    else
        zeros(n*(n+1)/2), zeros(n*(n+1)/2)
    end
    # Dual variables
    u, u_old = zeros(m+p), zeros(m+p)
    end

    # Diagonal scaling
    M = vcat(affine_sets.A, affine_sets.G)
    Mt = M'
    affine_sets, TMt, Tc, S, SM, Sinv = diag_scaling(affine_sets, 1.0, dims, M, Mt)

    # Stepsize parameters
    # L = 1.0 / svds(M; nsv=1)[1][:S][1]
    L = 1.0 / svds(SM; nsv=1)[1][:S][1]

    s0, t0 = sqrt(L), sqrt(L)
    s, t, adapt_decay, adapt_level, adapt_threshold = init_balance_stepsize(s0, t0)
    adapt_level2 = adapt_level
    stepsize_update = 0
    rank_update = 0

    # Initialization
    v0 = zeros((0, ))
    nev = 1 # Initial target-rank
    adapt_level2 = 0.9
    a = AllocatedData(dims)

    s, t = 1.0, 1.0

    # Fixed-point loop
    @timeit "CP loop" for k in 1:max_iter

        # Update primal variable
        @timeit "primal" primal_step!(x, u, a, dims, conic_sets, nev, v0, primal_tol, t, TMt, Tc)
        
        # Update dual variable
        @timeit "dual" dual_step!(x, u, x_old, a, dims, affine_sets, s, S, SM, Sinv)

        # Compute residuals
        @timeit "logging" begin
            push!(primal_residual, norm(x_old - x))
            push!(dual_residual, norm(u_old - u))
            push!(comb_residual, primal_residual[end] + dual_residual[end])
            if mod(k, 500) == 0 && opt.verbose
                print_progress(k, primal_residual[end], dual_residual[end])
            end
        end

        copy!(x_old, x)
        copy!(u_old, u)
    
        # Adaptive stepsizes
        if primal_residual[end] > 0.1 * dual_residual[end] * adapt_threshold
            t /= (1 - adapt_level)
            s *= (1 - adapt_level)
            adapt_level *= adapt_decay
        elseif primal_residual[end] < 0.1 * dual_residual[end] / adapt_threshold
            t *= (1 - adapt_level)
            s /= (1 - adapt_level)
            adapt_level *= adapt_decay    
        end
        if primal_residual[end] < primal_tol && dual_residual[end] > 10 * dual_tol
            t *= (1 - adapt_level2)
            s /= (1 - adapt_level2)
            adapt_level2 *= adapt_decay 
            stepsize_update = 0
        elseif primal_residual[end] > 10 * primal_tol && dual_residual[end] < dual_tol
            t /= (1 - adapt_level2)
            s *= (1 - adapt_level2)
            stepsize_update = 0
            adapt_level2 *= adapt_decay 
        end

        # Check convergence
        rank_update += 1
        if primal_residual[end] < primal_tol && dual_residual[end] < dual_tol
            # Check convergence of inexact fixed-point
            @timeit "primal" primal_step!(x, u, a, dims, conic_sets, nev + 1, v0, primal_tol, t, TMt, Tc)
            @timeit "dual" dual_step!(x, u, x_old, a, dims, affine_sets, s, S, SM, Sinv)

            push!(primal_residual, norm(x_old - x))
            push!(dual_residual, norm(u_old - u))
            push!(comb_residual, primal_residual[end] + dual_residual[end])
            print_progress(k, primal_residual[end], dual_residual[end])

            if primal_residual[end] < primal_tol && dual_residual[end] < dual_tol
                converged = true
                break
            else
                nev *= 2
                rank_update = 0
                println("Updating target-rank to. = $nev")
            end

        elseif k > 2000 && comb_residual[end - 1999] < comb_residual[end] && rank_update > 2000
            nev *= 2
            rank_update = 0
            print_progress(k, primal_residual[end], dual_residual[end])
            println("Updating target-rank to = $nev")
        end
    end

    time = toq()
    println("Time = $time")
    @show dot(c_orig, x)

    return CPResult(Int(converged), x, u, 0.0*x, primal_residual[end], dual_residual[end], dot(c_orig, x))
end

function initialize(x, u, dims::Dims, aff::AffineSets, t, Mt, M, s)
    x_old = zeros(dims.n*(dims.n+1)/2)
    iv = aff.sdpcone[1][1]::Vector{Int}
    im = aff.sdpcone[1][2]::Vector{Int}
    X = eye(dims.n)
    
    for i in 1:10
        x = x - t * (Mt * u) - t * aff.c
        for i in eachindex(iv)
            X[im[i]] = x[iv[i]]
        end
        X = max.(min.(X, 1.0), -1.0)
        for i in 1:dims.n
            X[i,i] = max(X[i,i], 0.0)
        end
        for i in eachindex(iv)
            x[iv[i]] = X[im[i]]
        end
        # x = max.(x - t * (Mt * u) - t * aff.c, 0.0)
        u += s * M * (2.0 * x - x_old)::Vector{Float64}
        u -= s * box_projection(u ./ s, dims, aff)::Vector{Float64}
        copy!(x_old, x)
    end

    return x, x_old, u
end

function diag_scaling(affine_sets, alpha, dims, M, Mt)
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

function init_balance_stepsize(s0, t0)::Tuple{Float64,Float64,Float64,Float64,Float64}
    adapt_level = 0.5      # Factor by which the stepsizes will be balanced 
    adapt_decay = 0.6     # Rate the adaptivity decreases over time
    adapt_threshold = 1.5  # Minimum value that trigger to recompute the stepsizes 
    return s0, t0, adapt_decay, adapt_level, adapt_threshold
end

function dual_step!(x, u, x_old, a, dims, affine_sets, s, S, SM, Sinv)
    copy!(a.x_1, x_old)
    Base.LinAlg.axpy!(-2.0, x, a.x_1) # alpha*x + y
    A_mul_B!(a.u_1, SM, a.x_1)
    Base.LinAlg.axpy!(-s, a.u_1, u) # alpha*x + y

    @inbounds @simd for i in eachindex(u)
        a.u_1[i] = Sinv[i] * u[i] / s
    end
    box_projection!(a.u_1, dims, affine_sets)
    A_mul_B!(a.u_2, S, a.u_1)
    Base.LinAlg.axpy!(-s, a.u_2, u) # alpha*x + y
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

function primal_step!(x, u, a, dims, conic_sets, nev, v0, eig_tol, t, TMt, Tc)::Void
    #x=x+(-t)*(TMt*u)+(-t)*Tc
    A_mul_B!(a.x_1, TMt, u) #(TMt*u)
    Base.LinAlg.axpy!(-t, a.x_1, x) # x=x+(-t)*(TMt*u)
    Base.LinAlg.axpy!(-t, Tc, x) # x=x+(-t)*Tc
    
    sdp_cone_projection!(x, a, dims, conic_sets, nev, v0, eig_tol)
end

function sdp_cone_projection!(v::Vector{Float64}, a::AllocatedData, dims::Dims, con::ConicSets, nev::Int64, v0::Vector{Float64}, eig_tol::Float64)::Void

    iv = con.sdpcone[1][1]::Vector{Int}
    im = con.sdpcone[1][2]::Vector{Int}
    @timeit "reshape" begin
        @inbounds @simd for i in eachindex(iv)
            a.m.data[im[i]] = v[iv[i]]
        end
    end

    @timeit "eig" begin
        if false
            fact = @timeit "syver" eigfact!(a.m, 0.0, Inf)
            a.m = fact[:vectors] * spdiagm(fact[:values]) * fact[:vectors]'
        else
            D, V = @timeit "arpack" eigs(a.m; nev=nev, which=:LR, maxiter=10000, tol=eig_tol, v0=v0)::Tuple{Array{Float64,1},Array{Float64,2},Int64,Int64,Int64,Array{Float64,1}}
            fill!(a.m.data, 0.0)
            for i in 1:min(nev, dims.n)
                if D[i] > 0.0
                    Base.LinAlg.BLAS.gemm!('N', 'T', D[i], V[:, i], V[:, i], 1.0, a.m.data)
                end
            end
            # a.m = Symmetric(V * spdiagm(max.(D, 0.0)) * V')
            # copy!(v0, V[:, 1])
        end
    end

    @timeit "reshape" begin
        @inbounds @simd for i in eachindex(iv)
            v[iv[i]] = a.m.data[im[i]]
        end
    end

    return nothing
end

function print_progress(k, primal_res, dual_res)
    s_k = @sprintf("%d",k)
    s_k *= " |"
    s_s = @sprintf("%.4f",primal_res + dual_res)
    s_s *= " |"
    s_p = @sprintf("%.4f",primal_res)
    s_p *= " |"
    s_d = @sprintf("%.4f",dual_res)
    s_d *= " |"
    a = "|"
    a *= " "^max(0,9-length(s_k))
    a *= s_k
    a *= " "^max(0,12-length(s_s))
    a *= s_s
    a *= " "^max(0,12-length(s_p))
    a *= s_p
    a *= " "^max(0,12-length(s_d))
    a *= s_d
    println(a)

end

end