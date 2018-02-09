module ProxSDP

using MathOptInterface, TimerOutputs
# using PyCall
# @pyimport scipy.linalg as sp
# @pyimport scipy.sparse.linalg as ppp

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

function chambolle_pock(
    affine_sets, dims, verbose=true, max_iter=Int(1e+5), primal_tol=1e-5, dual_tol=1e-5
)   
    opt = CPOptions(false, verbose)
    c_orig = zeros(1)
    # use algorithm in full square matrix mode or triangular mode
    if opt.fullmat
        M = zeros(Int, dims.n, dims.n)
        iv = affine_sets.sdpcone[1][1]
        im = affine_sets.sdpcone[1][2]
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
        iv = affine_sets.sdpcone[1][1]
        im = affine_sets.sdpcone[1][2]
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

    converged = false
    tic()
    @timeit "print" if verbose
        println(" Initializing Primal-Dual Hybrid Gradient method")
        println("----------------------------------------------------------")
        println("|  iter  | comb. res | prim. res |  dual res |    rho    |")
        println("----------------------------------------------------------")
    end

    @timeit "Init" begin
    # Given
    m, n, p = dims.m, dims.n, dims.p

    # logging
    best_comb_residual = Inf
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
    alpha = 1.0
    affine_sets, TMt, Tc, S, SM, Sinv = diag_scaling(affine_sets, alpha, dims)
    M = vcat(affine_sets.A, affine_sets.G)
    Mt = M'

    theta = 1.0
    L = 1.0 / norm(full(M'* M))
    s, t = sqrt(L), sqrt(L)

    # Fixed-point loop
    @timeit "CP loop" for k in 1:max_iter

        # Update primal variable
        @timeit "primal" begin
            x =  sdp_cone_projection(x - TMt * u - Tc, dims, affine_sets, opt, k, polishing)::Vector{Float64}
            # x = sdp_cone_projection(x - t * (Mt * u) - affine_sets.c, dims, affine_sets, opt, k)::Vector{Float64}
        end

        # Update dual variable
        @timeit "dual" begin
            u += SM * ((1.0 + theta) * x - x_old)::Vector{Float64}
            u -= S * box_projection(Sinv .* u, dims, affine_sets)::Vector{Float64}
            # u += s * M * ((1.0 + theta) * x - x_old)::Vector{Float64}
            # u -= s * box_projection(u ./ s, dims, affine_sets)::Vector{Float64}
        end

        # Compute residuals
        @timeit "logging" begin
            P = (1 / t) * (x_old - x) - Mt * (u_old - u)::Vector{Float64}
            D = (1 / s) * (u_old - u) - M * (x_old - x)::Vector{Float64}
            push!(primal_residual, norm(P))
            push!(dual_residual, norm(D))
            push!(comb_residual, primal_residual[end] + dual_residual[end])

            if mod(k, 100) == 0 && opt.verbose
                print_progress(k, primal_residual[end], dual_residual[end])
            end
        end

        # Keep track
        @timeit "tracking" begin
            copy!(x_old, x)
            copy!(u_old, u)
        end

        # Check convergence
        if primal_residual[end] < primal_tol && dual_residual[end] < dual_tol && polishing
            converged = true
            break
        elseif primal_residual[end] < 10 * primal_tol && dual_residual[end] < 10 * dual_tol && polishing == false
            println("Starting polishing procedure ---")
            polishing = true
        end
    end

    time = toq()
    println("Time = $time")

    @show u
    @show dot(c_orig, x)

    return CPResult(Int(converged), x, u, 0.0*x, primal_residual[end], dual_residual[end], dot(c_orig, x))
end

function diag_scaling(affine_sets, alpha, dims)
    # Build full problem matrices
    M = vcat(affine_sets.A, affine_sets.G)
    rhs = vcat(affine_sets.b, affine_sets.h)
    Mt = M'

    # Diagonal scaling
    div = vec(sum(abs.(M).^(2.0-alpha), 1))
    div[find(x-> x == 0.0, div)] = 1.0
    T = sparse(diagm(1.0 ./ div))
    div = vec(sum(abs.(M).^alpha, 2))
    div[find(x-> x == 0.0, div)] = 1.0
    S = sparse(diagm(1.0 ./ div))
    Sinv = div

    # Cache matrix multiplications
    TMt = T * Mt
    Tc = T * affine_sets.c
    SM = S * M
    Srhs = S * rhs
    affine_sets.b = Srhs[1:dims.p]
    affine_sets.h = Srhs[dims.p+1:end]

    return affine_sets, TMt, Tc, S, SM, Sinv
end

function box_projection(v::Vector{Float64}, dims::Dims, aff::AffineSets)::Vector{Float64}
    # Projection onto = b
    if !isempty(aff.b) 
        v[1:dims.p] = aff.b
    end
    # Projection onto <= h
    if !isempty(aff.h)
        v[dims.p+1:end] = min.(v[dims.p+1:end], aff.h)
    end
    return v
end

function sdp_cone_projection(v::Vector{Float64}, dims::Dims, aff::AffineSets, opt::CPOptions, iter, polishing)::Vector{Float64}

    if opt.fullmat
        X0 = @timeit "reshape" Symmetric(reshape(v, (dims.n, dims.n)))::Matrix{Float64}
        fact1 = @timeit "eig" eigfact!(X0, 0.0, Inf)
        M0 = fact1[:vectors] * diagm(fact1[:values]) * fact1[:vectors]' ::Matrix{Float64}
        v = vec(M0)::Vector{Float64}
    else
        @timeit "reshape" begin
            M = zeros(dims.n, dims.n)::Matrix{Float64}
            iv = aff.sdpcone[1][1]::Vector{Int}
            im = aff.sdpcone[1][2]::Vector{Int}
            for i in eachindex(iv)
                M[im[i]] = v[iv[i]]
            end
            X = Symmetric(M,:L)::Symmetric{Float64,Array{Float64,2}}
        end

        if polishing
            fact = @timeit "eig" eigfact!(X, 0.0, Inf)
            D = diagm(max.(fact[:values], 0.0))
            M3 = fact[:vectors] * D * fact[:vectors]'
            for i in eachindex(iv)
                v[iv[i]] = M3[im[i]]
            end
        elseif iter < 10
            fact = @timeit "eig" eigfact!(X, 0.0, Inf)
            M1 = fact[:vectors] * diagm(fact[:values]) * fact[:vectors]'::Matrix{Float64}
            for i in eachindex(iv)
                v[iv[i]] = M1[im[i]]
            end
        else
            D, V = @timeit "eig" eigs(X; nev=1, which=:LR)::Tuple{Array{Float64,1},Array{Float64,2},Int64,Int64,Int64,Array{Float64,1}}
            D2 = diagm(max.(D, 0.0))::Matrix{Float64}
            M2 = vec(V * D2 * V')::Vector{Float64}
            v0 = vec(copy(V))::Vector{Float64}
            for i in eachindex(iv)
                v[iv[i]] = M2[im[i]]
            end
        end

        # fact = @timeit "eig" eigfact!(X, 0.0, Inf)
        # D = diagm(max.(fact[:values], 0.0))
        # M2 = fact[:vectors] * D * fact[:vectors]'

        # for i in eachindex(iv)
        #     v[iv[i]] = M2[im[i]]
        # end
    end

    return v::Vector{Float64}
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