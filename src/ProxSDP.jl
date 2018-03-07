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

type PrimalDual
    x::Vector{Float64}
    x_old::Vector{Float64}
    u::Vector{Float64}
    u_old::Vector{Float64}
    PrimalDual(dims) = new(zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2), zeros(dims.m+dims.p), zeros(dims.m+dims.p))
end

type AuxiliaryData
    m::Symmetric{Float64,Matrix{Float64}}
    TMtu::Vector{Float64}
    TMtu_old::Vector{Float64}
    SMx::Vector{Float64}
    SMx_old::Vector{Float64}
    u_1::Vector{Float64}
    u_2::Vector{Float64}
    AuxiliaryData(dims) = new(
        Symmetric(zeros(dims.n, dims.n), :L), zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2),
        zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m)
    )
end

function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, dims::Dims, verbose=true, max_iter=Int(1e+5), primal_tol=1e-4, dual_tol=1e-4)::CPResult

    tic()
    println(" Initializing Primal-Dual Hybrid Gradient method")
    println("----------------------------------------------------------")
    println("|  iter  | comb. res | prim. res |  dual res |    rank   |")
    println("----------------------------------------------------------")

    @timeit "Init" begin
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

        # Initialization
        pair = PrimalDual(dims)
        a = AuxiliaryData(dims)
        target_rank = 1
        
        # logging
        converged = false
        rank_update = 0
        best_prim_residual, best_dual_residual = Inf, Inf
        converged, status, polishing = false, false, false
        # comb_residual, dual_residual, primal_residual = Float64[], Float64[], Float64[]
        # sizehint!(comb_residual, max_iter)
        # sizehint!(dual_residual, max_iter)
        # sizehint!(primal_residual, max_iter)
        primal_residual, dual_residual, comb_residual = zeros(max_iter), zeros(max_iter), zeros(max_iter)
    end

    @timeit "Scaling" begin
        # Diagonal scaling
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'
        affine_sets, TMt, Tc, S, SM, Sinv = diag_scaling(affine_sets, 1.0, dims, M, Mt)
    end

    # Stepsize parameters
    L = 1.0 / svds(M; nsv=1)[1][:S][1]
    primal_step, dual_step = sqrt(L), sqrt(L)  
    adapt_level = 0.5       # Factor by which the stepsizes will be balanced 
    adapt_decay = 0.9       # Rate the adaptivity decreases over time
    adapt_threshold = 1.5   # Minimum value that trigger to recompute the stepsizes

    # Fixed-point loop
    @timeit "CP loop" for k in 1:max_iter

        # Update primal variable
        @timeit "primal" target_rank = primal_step!(pair, a, dims, conic_sets, target_rank, TMt, Tc, primal_step)::Int64

        # Update dual variable
        @timeit "dual" dual_step!(pair, a, dims, affine_sets, S, SM, Sinv, dual_step)::Void

        # Compute residuals and update old iterates
        @timeit "logging" compute_residual(pair, a, primal_residual, dual_residual, comb_residual, primal_step, dual_step, k)::Void

        # Print progress
        if mod(k, 1000) == 0 && opt.verbose
            print_progress(k, primal_residual[k], dual_residual[k], target_rank)::Void
        end

        # Check convergence
        rank_update += 1
        if comb_residual[k] < 1e-3
            # Check convergence of inexact fixed-point
            @timeit "primal" target_rank = primal_step!(pair, a, dims, conic_sets, target_rank + 1, TMt, Tc, primal_step)::Int64
            @timeit "dual" dual_step!(pair, a, dims, affine_sets, S, SM, Sinv, dual_step)::Void
            @timeit "logging" compute_residual(pair, a, primal_residual, dual_residual, comb_residual, primal_step, dual_step, k)::Void
            print_progress(k, primal_residual[k], dual_residual[k], target_rank)::Void

            if comb_residual[k] < 1e-3
                converged = true
                best_prim_residual, best_dual_residual = primal_residual[k], dual_residual[k]
                break
            elseif rank_update > 1000
                target_rank *= 2
                rank_update = 0
                if target_rank < 9
                    adapt_level = 0.5 
                end
            end

        # Check divergence
        elseif k > 3000 && comb_residual[k - 2999] < comb_residual[k] && rank_update > 2000
            target_rank *= 2
            rank_update = 0
            if target_rank < 9
                adapt_level = 0.5 
            end
            print_progress(k, primal_residual[k], dual_residual[k], target_rank)::Void

        # Adaptive stepsizes
        elseif primal_residual[k] > 100 * primal_tol && dual_residual[k] < dual_tol 
            primal_step /= (1 - adapt_level)
            dual_step *= (1 - adapt_level)
            adapt_level *= adapt_decay
        elseif primal_residual[k] < primal_tol && dual_residual[k] > 100 * dual_tol
            primal_step *= (1 - adapt_level)
            dual_step /= (1 - adapt_level)
            adapt_level *= adapt_decay 
        end
    end

    time = toq()
    println("Time = $time")
    @show dot(c_orig, pair.x)

    return CPResult(Int(converged), pair.x, pair.u, 0.0*pair.x, best_prim_residual, best_dual_residual, dot(c_orig, pair.x))
end

function compute_residual(pair::PrimalDual, a::AuxiliaryData, primal_residual::Array{Float64,1}, dual_residual::Array{Float64,1}, comb_residual::Array{Float64,1}, primal_step::Float64, dual_step::Float64, iter::Int64)::Void    
    # Compute primal residual
    Base.LinAlg.axpy!(-1.0, a.TMtu, a.TMtu_old)
    Base.LinAlg.axpy!((1.0 / primal_step), pair.x_old, a.TMtu_old)
    Base.LinAlg.axpy!(-(1.0 / primal_step), pair.x, a.TMtu_old)
    primal_residual[iter] = norm(a.TMtu_old, 2)

    # Compute dual residual
    Base.LinAlg.axpy!(-1.0, a.SMx, a.SMx_old)
    Base.LinAlg.axpy!((1.0 / dual_step), pair.u_old, a.SMx_old)
    Base.LinAlg.axpy!(-(1.0 / dual_step), pair.u, a.SMx_old)
    dual_residual[iter] = norm(a.SMx_old, 2)

    # Compute combined residual
    comb_residual[iter] = primal_residual[iter] + dual_residual[iter]

    # Keep track of previous iterates
    copy!(pair.x_old, pair.x)
    copy!(pair.u_old, pair.u)
    copy!(a.TMtu_old, a.TMtu)
    copy!(a.SMx_old, a.SMx)
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

function dual_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, affine_sets::AffineSets, S::SparseMatrixCSC{Float64,Int64}, SM::SparseMatrixCSC{Float64,Int64}, Sinv::Vector{Float64}, dual_step::Float64)::Void
    copy!(a.u_1, a.SMx_old)
    A_mul_B!(a.SMx, SM, pair.x)
    Base.LinAlg.axpy!(-2.0, a.SMx, a.u_1) # alpha*x + y
    Base.LinAlg.axpy!(-dual_step, a.u_1, pair.u) # alpha*x + y

    @inbounds @simd for i in eachindex(pair.u)
        a.u_1[i] = Sinv[i] * pair.u[i] / dual_step
    end
    box_projection!(a.u_1, dims, affine_sets)
    A_mul_B!(a.u_2, S, a.u_1)
    Base.LinAlg.axpy!(-dual_step, a.u_2, pair.u) # alpha*x + y
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

function primal_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, conic_sets::ConicSets, target_rank::Int64, TMt::SparseMatrixCSC{Float64,Int64}, Tc::Vector{Float64}, primal_step::Float64)::Int64
    # x=x+(-t)*(TMt*u)+(-t)*Tc
    A_mul_B!(a.TMtu, TMt, pair.u) # (TMt*u)
    Base.LinAlg.axpy!(-primal_step, a.TMtu, pair.x) # x=x+(-t)*(TMt*u)
    Base.LinAlg.axpy!(-primal_step, Tc, pair.x) # x=x+(-t)*Tc

    # Projection onto the psd cone
    target_rank = sdp_cone_projection!(pair.x, a, dims, conic_sets, target_rank)::Int64
    return target_rank
end

function sdp_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, dims::Dims, con::ConicSets, target_rank::Int64)::Int64

    iv = con.sdpcone[1][1]::Vector{Int}
    im = con.sdpcone[1][2]::Vector{Int}
    @timeit "reshape" begin
        @inbounds @simd for i in eachindex(iv)
            a.m.data[im[i]] = v[iv[i]]
        end
    end

    if target_rank < 8
        try
            @timeit "eigs" begin
                D, V = eigs(a.m; nev=target_rank, which=:LR, maxiter=100000)::Tuple{Array{Float64,1},Array{Float64,2},Int64,Int64,Int64,Array{Float64,1}}
                fill!(a.m.data, 0.0)
                for i in 1:min(target_rank, dims.n)
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
            return target_rank
        end
    end
    
    @timeit "eigfact" begin
        fact = eigfact!(a.m, 0.0, Inf)
        fill!(a.m.data, 0.0)
        target_rank = length(fact[:values])
        for i in 1:target_rank
            Base.LinAlg.BLAS.gemm!('N', 'T', fact[:values][i], fact[:vectors][:, i], fact[:vectors][:, i], 1.0, a.m.data)
        end
    end

    @timeit "reshape" begin
        @inbounds @simd for i in eachindex(iv)
            v[iv[i]] = a.m.data[im[i]]
        end
    end
    return target_rank
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