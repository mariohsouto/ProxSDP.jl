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

    # theta::Float64
    # nev::Int

end

type AllocatedData
    m1::Symmetric{Float64,Matrix{Float64}}#Matrix{Float64}
    m2::Matrix{Float64}
    m3::Matrix{Float64}
    # m3::Symmetric{Int64,Matrix{Float64}}
    AllocatedData(dims) = new(Symmetric(zeros(dims.n, dims.n), :L),zeros(dims.n, dims.n),zeros(dims.n, dims.n))
end

function chambolle_pock(
    affine_sets, conic_sets, dims, verbose=true, max_iter=Int(1e+5), primal_tol=1e-4, dual_tol=1e-4
)   
    @timeit "Init" begin
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

        converged = false
        tic()
        println(" Initializing Primal-Dual Hybrid Gradient method")
        println("----------------------------------------------------------")
        println("|  iter  | comb. res | prim. res |  dual res |    rho    |")
        println("----------------------------------------------------------")

        # Given
        m, n, p = dims.m, dims.n, dims.p

        # logging
        best_comb_residual, comb_res_rank_update = Inf, Inf
        converged, status, polishing = false, false, false
        comb_residual, dual_residual, primal_residual = Float64[], Float64[], Float64[]
        sizehint!(comb_residual, max_iter)
        sizehint!(dual_residual, max_iter)
        sizehint!(primal_residual, max_iter)
        rank_updated = false
        rank_history = Float64[]
        sizehint!(rank_history, max_iter)

        # Primal variables
        x, x_old =  if opt.fullmat
            zeros(n^2), zeros(n^2)
        else
            zeros(n*(n+1)/2), zeros(n*(n+1)/2)
        end
        # Dual variables
        u, u_old = zeros(m+p), zeros(m+p)

        # Diagonal scaling
        alpha = 1.0
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'
        affine_sets, TMt, Tc, S, SM, Sinv = diag_scaling(affine_sets, alpha, dims, M, Mt)

        # Overrelaxation parameter
        theta = 1.0
        # Initial target-rank
        nev = 1
        # Initial stepsizes
        L = 1.0 / svds(M; nsv=1)[1][:S][1]
        s0, t0 = sqrt(L), sqrt(L)
        # Adaptive target rank parameters
        s, t, adapt_decay, adapt_level, adapt_threshold = init_balance_stepsize(s0, t0)

        # Initialization
        for i in 1:100
            x = max.(x - t * TMt * u - t * Tc, 0.0)
            u += s * SM * ((1.0 + theta) * x - x_old)::Vector{Float64}
            u -= s * S * box_projection((Sinv .* u) ./ s, dims, affine_sets)::Vector{Float64}
        end
        for i in 1:100
            x = rank_one_projection(x - t * TMt * u - t * Tc, dims, affine_sets, conic_sets)
            u += s * SM * ((1.0 + theta) * x - x_old)::Vector{Float64}
            u -= s * S * box_projection((Sinv .* u) ./ s, dims, affine_sets)::Vector{Float64}
        end

        a = AllocatedData(dims)
    end
    
    TMtu = similar(x)
    u_temp = similar(u)
    u_temp2 = similar(u)
    x_temp = similar(x)

    # Fixed-point loop
    @timeit "CP loop" for k in 1:1501#max_iter

        # Update primal variable
        @timeit "primal" begin
            # x = sdp_cone_projection(x - t * (Mt * u) - t * affine_sets.c, dims, affine_sets, opt, k, polishing, nev)::Vector{Float64}
            # x = x - t * TMt * u - t * Tc
            A_mul_B!(TMtu,TMt,u)
            Base.LinAlg.axpy!(-t,TMtu,x) # alpha*x + y
            Base.LinAlg.axpy!(-t,Tc,x) # alpha*x + y
            sdp_cone_projection!(x, a, dims, conic_sets, opt, k, polishing, nev, rank_history)#::Vector{Float64}
        end

        # Update dual variable
        @timeit "dual" begin
            # u += s * SM * ((1.0 + theta) * x - x_old)::Vector{Float64}
            # u += - s * M * (-(1.0 + theta) * x + x_old)::Vector{Float64}
            # u -= s * box_projection(u ./ s, dims, affine_sets)::Vector{Float64}
            # u -= s * S * u_temp::Vector{Float64}

            # u += - s * M * (-(1.0 + theta) * x + x_old)::Vector{Float64}
            copy!(x_temp,x_old)
            Base.LinAlg.axpy!(-(1.0 + theta),x,x_temp) # alpha*x + y
            A_mul_B!(u_temp,SM,x_temp)
            Base.LinAlg.axpy!(-s,u_temp,u) # alpha*x + y
            # u -= s * S * box_projection((Sinv .* u) ./ s, dims, affine_sets)::Vector{Float64}
            # u_temp .= (Sinv .* u) ./ s
            @inbounds @simd for i in eachindex(u)
                u_temp[i] = Sinv[i] *u[i] *s
            end
            box_projection!(u_temp, dims, affine_sets)
            A_mul_B!(u_temp2,S,u_temp)
            Base.LinAlg.axpy!(-s,u_temp2,u) # alpha*x + y
        end

        # Compute residuals
        @timeit "logging" begin
            if mod(k, 500) == 0# && opt.verbose
                push!(primal_residual, norm((1 / t) * (x_old - x) - Mt * (u_old - u)))
                push!(dual_residual, norm((1 / s) * (u_old - u) - M * (x_old - x)))
                push!(comb_residual, primal_residual[end] + dual_residual[end])
                if opt.verbose
                    print_progress(k, primal_residual[end], dual_residual[end])
                end
            else 
                push!(primal_residual, -1)
                push!(dual_residual, -1)
                push!(comb_residual, -1)
            end
        end
            
        # Adaptive stepsizes
        if mod(k, 500) == 0
            if primal_residual[end] > dual_residual[end] * adapt_threshold
                t /= (1 - adapt_level)
                s *= (1 - adapt_level)
                adapt_level *= adapt_decay
            elseif primal_residual[end] < dual_residual[end] / adapt_threshold
                t *= (1 - adapt_level)
                s /= (1 - adapt_level)
                adapt_level *= adapt_decay    
            end
        end

        # Keep track
        @timeit "tracking" begin
            copy!(x_old, x)
            copy!(u_old, u)
        end

        # Check convergence
        if mod(k, 500) == 0
            if primal_residual[end] < primal_tol && dual_residual[end] < dual_tol # && rank_updated
                println("lower bound = $(sqrt(n) / k))")
                converged = true
                break
            end
        end
    end

    time = toq()
    println("Time = $time")

    @show dot(c_orig, x)

    @show rank_history

    return CPResult(Int(converged), x, u, 0.0*x, primal_residual[end], dual_residual[end], dot(c_orig, x))
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
    adapt_level = 0.9     # Factor by which the stepsizes will be balanced 
    adapt_decay = 0.95    # Rate the adaptivity decreases over time
    adapt_threshold = 1.5 # Minimum value that trigger to recompute the stepsizes 
    return s0, t0, adapt_decay, adapt_level, adapt_threshold
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

function rank_one_projection(v::Vector{Float64}, dims::Dims, aff::AffineSets, con::ConicSets)
    M = zeros(dims.n, dims.n)::Matrix{Float64}
    iv = con.sdpcone[1][1]::Vector{Int}
    im = con.sdpcone[1][2]::Vector{Int}
    for i in eachindex(iv)
        M[im[i]] = v[iv[i]]
    end
    X = Symmetric(M, :L)::Symmetric{Float64,Array{Float64,2}}

    D, V = @timeit "eig" eigs(X; nev=1, which=:LR)::Tuple{Array{Float64,1},Array{Float64,2},Int64,Int64,Int64,Array{Float64,1}}
    M = vec(V * spdiagm(max.(D, 0.0)) * V')::Vector{Float64}
    for i in eachindex(iv)
        v[iv[i]] = M[im[i]]
    end

    return v
end

function sdp_cone_projection!(v::Vector{Float64}, a::AllocatedData, dims::Dims, con::ConicSets, opt::CPOptions, iter, polishing, nev, rank_history)#::Vector{Float64}

    if opt.fullmat
        # fact1 = @timeit "eig" eigfact!(Symmetric(reshape(v, (dims.n, dims.n)))::Matrix{Float64}, 0.0, Inf)
        fact1 = eigfact!(Symmetric(reshape(v, (dims.n, dims.n)))::Matrix{Float64}, 0.0, Inf)
        v = vec(fact1[:vectors] * diagm(fact1[:values]) * fact1[:vectors]')::Vector{Float64}
    else

        iv = con.sdpcone[1][1]::Vector{Int}
        im = con.sdpcone[1][2]::Vector{Int}
        @timeit "reshape" begin
            # M = zeros(dims.n, dims.n)::Matrix{Float64}
            @inbounds @simd for i in eachindex(iv)
                a.m1.data[im[i]] = v[iv[i]]
            end
            # sm1 = Symmetric(a.m1, :L)::Symmetric{Float64,Array{Float64,2}}
        end

        @timeit "eig" begin
            fact = @timeit "ineig" eigfact!(a.m1, sqrt(dims.n) / iter, Inf)
            # fact = @timeit "eig" eigfact!(X, 0.0, Inf)
            M2 = fact[:vectors] * spdiagm(fact[:values]) #* fact[:vectors]'
            # A_mul_B!(a.m2,fact[:vectors],spdiagm(fact[:values]))
            A_mul_Bt!(a.m3,M2,fact[:vectors])
            @inbounds @simd for i in eachindex(iv)
                v[iv[i]] = a.m3[im[i]]
            end
        end
    end

    if false
        push!(rank_history, length(fact[:values][fact[:values] .> 1e-5]))
    end

    # return v::Vector{Float64}, rank_history
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