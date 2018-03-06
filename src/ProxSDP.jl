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
    m::Symmetric{Float64,Matrix{Float64}}
    x_1::Vector{Float64}
    u_1::Vector{Float64}
    u_2::Vector{Float64}
    AllocatedData(dims) = new(Symmetric(zeros(dims.n, dims.n), :L), zeros(dims.n*(dims.n+1)/2), zeros(dims.p+dims.m), zeros(dims.p+dims.m))
end

function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, dims::Dims, verbose=true, max_iter=Int(1e+5), primal_tol=1e-4, dual_tol=1e-4)::CPResult

    opt = CPOptions(false, verbose)
    c_orig = zeros(1)

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

    converged = false
    tic()
    println(" Initializing Primal-Dual Hybrid Gradient method")
    println("----------------------------------------------------------")
    println("|  iter  | comb. res | prim. res |  dual res |    rank   |")
    println("----------------------------------------------------------")

    @timeit "Init" begin
        # Given
        m, n, p = dims.m, dims.n, dims.p

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

    # Initialization
    nev = 1 # Initial target-rank
    a = AllocatedData(dims)
    rank_update, adapt_stepsize = 0, 0

    # Stepsize parameters
    L = 1.0 / svds(M; nsv=1)[1][:S][1]
    primal_step, dual_step = sqrt(L), sqrt(L)  
    # primal_step, dual_step = 1.0, 1.0 # Initial stepsizes
    adapt_level = 0.5                 # Factor by which the stepsizes will be balanced 
    adapt_decay = 0.95                # Rate the adaptivity decreases over time
    adapt_threshold = 1.5             # Minimum value that trigger to recompute the stepsizes 

    # Fixed-point loop
    @timeit "CP loop" for k in 1:max_iter

        # Update primal variable
        @timeit "primal" primal_step!(x, u, a, dims, conic_sets, nev, TMt, Tc, primal_step)

        # Update dual variable
        @timeit "dual" dual_step!(x, u, x_old, a, dims, affine_sets, S, SM, Sinv, dual_step)

        # Compute residuals
        @timeit "logging" compute_residual(x, x_old, u, u_old, SM, TMt, primal_residual, dual_residual, comb_residual, primal_step, dual_step)

        # Print progress
        if mod(k, 1000) == 0 && opt.verbose
            print_progress(k, primal_residual[end], dual_residual[end], nev)
        end

        copy!(x_old, x)
        copy!(u_old, u)

        # Check convergence
        rank_update += 1
        adapt_stepsize += 1
        if comb_residual[end] < 1e-3
            # Check convergence of inexact fixed-point
            @timeit "primal" primal_step!(x, u, a, dims, conic_sets, nev + 1, TMt, Tc, primal_step)
            @timeit "dual" dual_step!(x, u, x_old, a, dims, affine_sets, S, SM, Sinv, dual_step)
            @timeit "logging" compute_residual(x, x_old, u, u_old, M, Mt, primal_residual, dual_residual, comb_residual, primal_step, dual_step)
            print_progress(k, primal_residual[end], dual_residual[end], nev)

            if comb_residual[end] < 1e-3
                converged = true
                break
            elseif adapt_stepsize > 2000
                nev *= 2
                rank_update = 0
                println("Updating target-rank to. = $nev")
            end

        # Check divergence
        elseif k > 2000 && comb_residual[end - 1999] < comb_residual[end] + 1e-5 && rank_update > 2000 && adapt_stepsize > 2000
            nev *= 2
            rank_update = 0
            print_progress(k, primal_residual[end], dual_residual[end], nev)
            println("Updating target-rank to = $nev")

        # Adaptive stepsizes
        elseif primal_residual[end] > 10 * primal_tol && dual_residual[end] < dual_tol
            primal_step /= (1 - adapt_level)
            dual_step *= (1 - adapt_level)
            adapt_level *= adapt_decay
        elseif primal_residual[end] < primal_tol && dual_residual[end] > 10 * dual_tol
            primal_step *= (1 - adapt_level)
            dual_step /= (1 - adapt_level)
            adapt_level *= adapt_decay   
        end
    end

    time = toq()
    println("Time = $time")
    @show dot(c_orig, x)

    return CPResult(Int(converged), x, u, 0.0*x, primal_residual[end], dual_residual[end], dot(c_orig, x))
end

function compute_residual(x::Vector{Float64}, x_old::Vector{Float64}, u::Vector{Float64}, u_old::Vector{Float64}, M::SparseMatrixCSC{Float64,Int64}, Mt::SparseMatrixCSC{Float64,Int64}, primal_residual::Array{Float64,1}, dual_residual::Array{Float64,1}, comb_residual::Array{Float64,1}, primal_step::Float64, dual_step::Float64)::Void    
    push!(primal_residual, norm((1.0 / primal_step) * (x - x_old) + Mt * (u - u_old)))
    push!(dual_residual, norm((1.0 / dual_step) * (u - u_old) + M * (x - x_old)))
    push!(comb_residual, primal_residual[end] + dual_residual[end])
    return nothing
end

function initialize(x::Vector{Float64}, u::Vector{Float64}, dims::Dims, aff::AffineSets, conic_sets::ConicSets, Mt::SparseMatrixCSC{Float64,Int64}, M::SparseMatrixCSC{Float64,Int64}, s::Float64, a::AllocatedData)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    x_old = zeros(dims.n*(dims.n+1)/2)
    iv = conic_sets.sdpcone[1][1]::Vector{Int}
    im = conic_sets.sdpcone[1][2]::Vector{Int}
    X = eye(dims.n)
    
    for i in 1:100
        A_mul_B!(a.x_1, Mt, u) #(TMt*u)
        Base.LinAlg.axpy!(-t, a.x_1, x) # x=x+(-t)*(TMt*u)
        Base.LinAlg.axpy!(-t, aff.c, x) # x=x+(-t)*Tc        
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
        copy!(a.x_1, x_old)
        Base.LinAlg.axpy!(-2.0, x, a.x_1) # alpha*x + y
        A_mul_B!(a.u_1, M, a.x_1)
        Base.LinAlg.axpy!(-s, a.u_1, u) # alpha*x + y
    
        @inbounds @simd for i in eachindex(u)
            a.u_1[i] = u[i] / s
        end
        box_projection!(a.u_1, dims, aff)
        Base.LinAlg.axpy!(-s, a.u_1, u) # alpha*x + y
        copy!(x_old, x)
    end

    return x, x_old, u
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

function dual_step!(x::Vector{Float64}, u::Vector{Float64}, x_old::Vector{Float64}, a::AllocatedData, dims::Dims, affine_sets::AffineSets, S::SparseMatrixCSC{Float64,Int64}, SM::SparseMatrixCSC{Float64,Int64}, Sinv::Vector{Float64}, dual_step::Float64)::Void
    copy!(a.x_1, x_old)
    Base.LinAlg.axpy!(-2.0, x, a.x_1) # alpha*x + y
    A_mul_B!(a.u_1, SM, a.x_1)
    Base.LinAlg.axpy!(-dual_step, a.u_1, u) # alpha*x + y

    @inbounds @simd for i in eachindex(u)
        a.u_1[i] = Sinv[i] * u[i] / dual_step
    end
    box_projection!(a.u_1, dims, affine_sets)
    A_mul_B!(a.u_2, S, a.u_1)
    Base.LinAlg.axpy!(-dual_step, a.u_2, u) # alpha*x + y

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

function primal_step!(x::Vector{Float64}, u::Vector{Float64}, a::AllocatedData, dims::Dims, conic_sets::ConicSets, nev::Int64, TMt::SparseMatrixCSC{Float64,Int64}, Tc::Vector{Float64}, primal_step::Float64)::Void
    # x=x+(-t)*(TMt*u)+(-t)*Tc
    A_mul_B!(a.x_1, TMt, u) # (TMt*u)
    Base.LinAlg.axpy!(-primal_step, a.x_1, x) # x=x+(-t)*(TMt*u)
    Base.LinAlg.axpy!(-primal_step, Tc, x) # x=x+(-t)*Tc

    # Projection onto the psd cone
    sdp_cone_projection!(x, a, dims, conic_sets, nev)::Void
    return nothing
end

function sdp_cone_projection!(v::Vector{Float64}, a::AllocatedData, dims::Dims, con::ConicSets, nev::Int64)::Void

    iv = con.sdpcone[1][1]::Vector{Int}
    im = con.sdpcone[1][2]::Vector{Int}
    @timeit "reshape" begin
        @inbounds @simd for i in eachindex(iv)
            a.m.data[im[i]] = v[iv[i]]
        end
    end

    if nev < 8
        try
            @timeit "eigs" begin
                D, V = eigs(a.m; nev=nev, which=:LR, maxiter=100000)::Tuple{Array{Float64,1},Array{Float64,2},Int64,Int64,Int64,Array{Float64,1}}
                fill!(a.m.data, 0.0)
                for i in 1:min(nev, dims.n)
                    if D[i] > 1e-6
                        Base.LinAlg.BLAS.gemm!('N', 'T', D[i], V[:, i], V[:, i], 1.0, a.m.data)
                    end
                end
            end
            @timeit "reshape" begin
                @inbounds @simd for i in eachindex(iv)
                    v[iv[i]] = a.m.data[im[i]]
                end
            end
            return nothing
        end
    end
    
    @timeit "eigfact" begin
        fact = eigfact!(a.m, 1e-6, Inf)
        fill!(a.m.data, 0.0)
        for i in 1:length(fact[:values])
            Base.LinAlg.BLAS.gemm!('N', 'T', fact[:values][i], fact[:vectors][:, i], fact[:vectors][:, i], 1.0, a.m.data)
        end
    end

    @timeit "reshape" begin
        @inbounds @simd for i in eachindex(iv)
            v[iv[i]] = a.m.data[im[i]]
        end
    end
    return nothing
end

function print_progress(k::Int64, primal_res::Float64, dual_res::Float64, nev::Int64)::Void
    s_k = @sprintf("%d", k)
    s_k *= " |"
    s_s = @sprintf("%.4f", primal_res + dual_res)
    s_s *= " |"
    s_p = @sprintf("%.4f", primal_res)
    s_p *= " |"
    s_d = @sprintf("%.4f", dual_res)
    s_d *= " |"
    s_nev = @sprintf("%.0f", nev)
    s_nev *= " |"
    a = "|"
    a *= " "^max(0, 9 - length(s_k))
    a *= s_k
    a *= " "^max(0, 12 - length(s_s))
    a *= s_s
    a *= " "^max(0, 12 - length(s_p))
    a *= s_p
    a *= " "^max(0, 12 - length(s_d))
    a *= s_d
    a *= " "^max(0, 12 - length(s_nev))
    a *= s_nev
    println(a)
    return nothing
end

end