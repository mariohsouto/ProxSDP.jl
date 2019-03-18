module ProxSDP

using MathOptInterface, TimerOutputs
using Arpack
using Compat
using Printf

include("MOI_wrapper.jl")
include("eigsolver.jl")
include("convex_sets.jl")
include("types.jl")
include("printing.jl")
include("chambolle_pock.jl")


MOIU.@model _ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)

Solver(;args...) = MOIU.CachingOptimizer(_ProxSDPModelData{Float64}(), ProxSDP.Optimizer(args))

function get_solution(opt::MOIU.CachingOptimizer{Optimizer,T}) where T
    return opt.optimizer.sol
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

function convergedrank(p::Params, cones::ConicSets, opt::Options)
    for (idx, sdp) in enumerate(cones.sdpcone)
        if !(p.min_eig[idx] < opt.tol_psd || p.target_rank[idx] > opt.max_target_rank_krylov_eigs || sdp.sq_side < opt.min_size_krylov_eigs)
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

function compute_residual!(pair::PrimalDual, a::AuxiliaryData, primal_residual::CircularVector{Float64}, dual_residual::CircularVector{Float64}, comb_residual::CircularVector{Float64}, mat::Matrices, p::Params)
    # Compute primal residual
    a.Mty_old .+= .- a.Mty .+ (1.0 / (1.0 + p.primal_step)) .* (pair.x_old .- pair.x)
    primal_residual[p.iter] = norm(a.Mty_old, 2) / (1.0 + max(p.norm_c, maximum(abs.(a.Mty))))
    # primal_residual[p.iter] = norm(a.Mty_old, 2) / (1.0 + p.norm_c)


    # Compute dual residual
    a.Mx_old .+= .- a.Mx .+ (1.0 / (1.0 + p.dual_step)) .* (pair.y_old .- pair.y)
    dual_residual[p.iter] = norm(a.Mx_old, 2) / (1.0 + max(p.norm_rhs, maximum(abs.(a.Mx))))
    # dual_residual[p.iter] = norm(a.Mx_old, 2) / (1.0 + p.norm_rhs)

    # Compute combined residual
    comb_residual[p.iter] = primal_residual[p.iter] + dual_residual[p.iter]

    # Keep track of previous iterates
    copyto!(pair.x_old, pair.x)
    copyto!(pair.y_old, pair.y)
    copyto!(a.Mty_old, a.Mty)
    copyto!(a.Mx_old, a.Mx)

    return nothing
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, affine_sets::AffineSets, mat::Matrices, opt::Options, p::Params)
    delta = .999
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
            copyto!(a.y_temp, a.y_half)
            box_projection!(a.y_half, affine_sets, p.beta * p.primal_step)
            a.y_temp .-= (p.beta * p.primal_step) .* a.y_half
        end

        @timeit "linesearch 3" mul!(a.Mty, mat.Mt, a.y_temp)
        
        # In-place norm
        @timeit "linesearch 4" begin
            a.Mty .-= a.Mty_old
            a.y_temp .-= pair.y_old
            y_norm = norm(a.y_temp)
            Mty_norm = norm(a.Mty)
        end

        if sqrt(p.beta) * p.primal_step * Mty_norm <= delta * y_norm
            break
        else
            p.primal_step *= 0.95
        end
    end

    # Reverte in-place norm
    a.Mty .+= a.Mty_old
    a.y_temp .+= pair.y_old

    copyto!(pair.y, a.y_temp)
    p.primal_step_old = p.primal_step

    return nothing
end

function preprocess!(aff::AffineSets, conic_sets::ConicSets)
    c_orig = zeros(1)
    if length(conic_sets.sdpcone) >= 1 || length(conic_sets.socone) >= 1
        all_cone_vars = Int[]
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
            append!(all_cone_vars, sdp_vars)
        end
        for (idx, soc) in enumerate(conic_sets.socone)
            soc_vars = copy(soc.idx)
            append!(all_cone_vars, soc_vars)
        end

        totvars = aff.n
        extra_vars = sort(collect(setdiff(Set(collect(1:totvars)),Set(all_cone_vars))))
        ord = vcat(all_cone_vars, extra_vars)
    else
        ord = collect(1:aff.n)
    end

    c_orig = copy(aff.c)

    aff.A, aff.G, aff.c = aff.A[:, ord], aff.G[:, ord], aff.c[ord]
    return c_orig[ord], sortperm(ord)
end

function map_socs!(v::Vector{Float64}, conic_sets::ConicSets, a::AuxiliaryData)
    cont = 0
    for (idx, sdp) in enumerate(conic_sets.sdpcone)
        cont += div(sdp.sq_side*(sdp.sq_side+1), 2)
    end
    sizehint!(a.soc_v, length(conic_sets.socone))
    sizehint!(a.soc_s, length(conic_sets.socone))
    for (idx, soc) in enumerate(conic_sets.socone)
        len = soc.len
        # push!(a.soc_v, view(v, cont+1:cont+len-1))
        # push!(a.soc_s, view(v, cont+len))
        push!(a.soc_s, view(v, cont+1))
        push!(a.soc_v, view(v, cont+2:cont+len))
        cont += len
    end
    return nothing
end

function primal_step!(pair::PrimalDual, a::AuxiliaryData, cones::ConicSets, mat::Matrices, arc::Vector{ARPACKAlloc{Float64}}, opt::Options, p::Params)

    pair.x .-= p.primal_step .* (a.Mty .+ mat.c)

    # Projection onto the psd cone
    if length(cones.sdpcone) >= 1
        @timeit "sdp proj" sdp_cone_projection!(pair.x, a, cones, arc, opt, p)
    end

    if length(cones.socone) >= 1
        @timeit "soc proj" so_cone_projection!(pair.x, a, cones, opt, p)
    end

    @timeit "linesearch -1" mul!(a.Mx, mat.M, pair.x)

    return nothing
end


function sdp_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, arc::Vector{ARPACKAlloc{Float64}}, opt::Options, p::Params)

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
        elseif !opt.full_eig_decomp && p.target_rank[idx] <= opt.max_target_rank_krylov_eigs && sdp.sq_side > opt.min_size_krylov_eigs
            @timeit "eigs" begin 
                eig!(arc[idx], a.m[idx], p.target_rank[idx], p.iter)
                if hasconverged(arc[idx])
                    fill!(a.m[idx].data, 0.0)
                    for i in 1:p.target_rank[idx]
                        if unsafe_getvalues(arc[idx])[i] > 0.0
                            current_rank += 1
                            vec = unsafe_getvectors(arc[idx])[:, i]
                            LinearAlgebra.BLAS.gemm!('N', 'T', unsafe_getvalues(arc[idx])[i], vec, vec, 1.0, a.m[idx].data)
                        end
                    end
                end
            end
            if hasconverged(arc[idx])
                @timeit "get min eig" p.min_eig[idx] = minimum(unsafe_getvalues(arc[idx]))
            else
                @timeit "eigfact" full_eig!(a, idx, opt)
            end
        else
            p.min_eig[idx] = 0.0
            @timeit "eigfact" full_eig!(a, idx, opt)
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

function full_eig!(a::AuxiliaryData, idx::Int, opt::Options)
    current_rank = 0
    # fact = eigen!(a.m[1], 1e-6, Inf)
    fact = eigen!(a.m[idx])
    fill!(a.m[idx].data, 0.0)
    for i in 1:length(fact.values)
        if fact.values[i] > 0.0
            current_rank += 1
            LinearAlgebra.BLAS.gemm!('N', 'T', fact.values[i], fact.vectors[:, i], fact.vectors[:, i], 1.0, a.m[idx].data)
        end
    end
    return nothing
end

function so_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, opt::Options, p::Params)
    for (idx, soc) in enumerate(cones.socone)
        # @show "a", pair.x
        soc_projection!(a.soc_v[idx], a.soc_s[idx])
        # @show "b", pair.x
    end
    return nothing
end

function soc_projection!(v::ViewVector, s::ViewScalar)
    nv = norm(v)
    if nv <= -s[]
        s[] = 0.0
        v .= 0.0
    elseif nv <= s[]
        #do nothing
    else
        val = 0.5 * (1.0+s[]/nv)
        v .*= val
        s[] = val * nv
    end
    return nothing
end

function soc_gap(v::ViewVector, s::ViewScalar)
    return norm(v) - s[]
end

function soc_convergence(a::AuxiliaryData, cones::ConicSets, pair::PrimalDual, opt::Options, p::Params)
    for (idx, soc) in enumerate(cones.socone)
        if soc_gap(a.soc_v[idx], a.soc_s[idx]) >= opt.tol_soc
            return false
        end
    end
    return true
end
end