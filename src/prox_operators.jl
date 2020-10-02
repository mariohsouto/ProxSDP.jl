function psd_vec_to_square(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, sqrt_2::Float64 = sqrt(2))
    # Build symmetric matrix(es) X
    @timeit "reshape1" begin
        cont = 1
        @inbounds for (idx, sdp) in enumerate(cones.sdpcone), j in 1:sdp.sq_side, i in 1:j#j:sdp.sq_side
            if i != j
                a.m[idx].data[i,j] = v[cont] / sqrt_2
            else
                a.m[idx].data[i,j] = v[cont]
            end
            # a.m[idx].data[i,j] = ifelse(i != j, v[cont] / sqrt_2, v[cont])
            cont += 1
        end
    end
    return cont
end
function psd_square_to_vec(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, sqrt_2::Float64 = sqrt(2))
    # Build symmetric matrix(es) X
    @timeit "reshape2" begin
        cont = 1
        @inbounds for (idx, sdp) in enumerate(cones.sdpcone), j in 1:sdp.sq_side, i in 1:j#j:sdp.sq_side
            if i != j
                v[cont] = a.m[idx].data[i, j] * sqrt_2
            else
                v[cont] = a.m[idx].data[i, j]
            end
            cont += 1
        end
    end
    return cont
end

function psd_projection!(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, opt::Options, p::Params, arc_list, iter::Int64)::Nothing

    p.min_eig, current_rank, sqrt_2 = zeros(length(cones.sdpcone)), 0, sqrt(2.)

    psd_vec_to_square(v, a, cones)

    # Project onto the p.s.d. cone
    for (idx, sdp) in enumerate(cones.sdpcone)
        p.current_rank[idx] = 0

        if sdp.sq_side == 1
            a.m[idx][1] = max(0., a.m[idx][1])
            p.min_eig[idx] = a.m[idx][1]
        elseif !opt.full_eig_decomp &&
                p.target_rank[idx] <= opt.max_target_rank_krylov_eigs &&
                sdp.sq_side > opt.min_size_krylov_eigs &&
                mod(p.iter, opt.full_eig_freq) > opt.full_eig_len # for full from time to time
            @timeit "eigs" if opt.eigsolver == 1
                arpack_eig!(arc_list[idx], a, idx, opt, p)
            else
                krylovkit_eig!(arc_list[idx], a, idx, opt, p)
            end
            if !hasconverged(arc_list[idx])
                @timeit "eigfact" full_eig!(a, idx, opt, p)
            end
        else
            @timeit "eigfact" full_eig!(a, idx, opt, p)
        end
    end

    psd_square_to_vec(v, a, cones)

    return nothing
end

function arpack_eig!(solver::EigSolverAlloc, a::AuxiliaryData, idx::Int, opt::Options, p::Params)::Nothing
    arpack_eig!(solver, a.m[idx], p.target_rank[idx], opt)
    if hasconverged(solver)
        fill!(a.m[idx].data, 0.)
        # TODO: how to measure this when convergen eigs is less than target?
        p.min_eig[idx] = minimum(arpack_getvalues(solver))
        # if solver.converged_eigs < p.target_rank[idx] && p.min_eig[idx] > 0.0
        #     p.min_eig[idx] = -Inf
        # end
        for i in 1:p.target_rank[idx]
            val = arpack_getvalues(solver)[i]
            if val > 0.
                p.current_rank[idx] += 1
                vec = view(arpack_getvectors(solver), :, i)
                LinearAlgebra.BLAS.gemm!('N', 'T', val, vec, vec, 1., a.m[idx].data)
            end
        end
    end
    return nothing
end

function krylovkit_eig!(solver::EigSolverAlloc, a::AuxiliaryData, idx::Int, opt::Options, p::Params)::Nothing
    krylovkit_eig!(solver, a.m[idx], p.target_rank[idx], opt)
    if hasconverged(solver)
        fill!(a.m[idx].data, 0.)
        # TODO: how to measure this when convergen eigs is less than target?
        # ? min_eig is just checking if the rank projection is going far enough to reach zeros and negatives
        p.min_eig[idx] = minimum(krylovkit_getvalues(solver))
        # if solver.converged_eigs < p.target_rank[idx] #&& p.min_eig[idx] > 0.0
        #     p.min_eig[idx] = -Inf
        # end
        for i in 1:min(p.target_rank[idx], solver.converged_eigs)
            val = krylovkit_getvalues(solver)[i]
            if val > 0.
                p.current_rank[idx] += 1
                vec = krylovkit_getvector(solver, i)
                LinearAlgebra.BLAS.gemm!('N', 'T', val, vec, vec, 1., a.m[idx].data)
            end
        end
    end
    return nothing
end

function full_eig!(a::AuxiliaryData, idx::Int, opt::Options, p::Params)::Nothing
    p.current_rank[idx] = 0
    fact = eigen!(a.m[idx])
    p.min_eig[idx] = 0.0 #minimum(fact.values)
    fill!(a.m[idx].data, 0.)
    for i in 1:length(fact.values)
        if fact.values[i] > 0.
            v = view(fact.vectors, :, i)
            LinearAlgebra.BLAS.gemm!('N', 'T', fact.values[i], v, v, 1., a.m[idx].data)
            if fact.values[i] > opt.tol_psd
                p.current_rank[idx] += 1
            end
        end
    end
    return nothing
end

function min_eig(a::AuxiliaryData, idx::Int, p::Params)
    # if p.min_eig[idx] == -Inf
    #     @timeit "bad min eig" begin
    #         fact = eigen!(a.m[idx])
    #         p.min_eig[idx] = minimum(fact.values)
    #     end
    # end
    return p.min_eig[idx]
end

function soc_projection!(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, opt::Options, p::Params)::Nothing
    for (idx, soc) in enumerate(cones.socone)
        soc_projection!(a.soc_v[idx], a.soc_s[idx])
    end
    return nothing
end

function soc_projection!(v::ViewVector, s::ViewScalar)::Nothing
    nv = norm(v, 2)
    if nv <= -s[]
        s[] = 0.
        v .= 0.
    elseif nv <= s[]
        #do nothing
    else
        val = .5 * (1. + s[] / nv)
        v .*= val
        s[] = val * nv
    end
    return nothing
end

function box_projection!(v::Array{Float64,1}, aff::AffineSets, step::Float64)::Nothing
    # Projection onto = b
    @simd for i in 1:length(aff.b)
        @inbounds v[i] = aff.b[i]
    end
    # Projection onto <= h
    @simd for i in 1:length(aff.h)
        @inbounds v[aff.p+i] = min(v[aff.p+i] / step, aff.h[i])
    end
    return nothing
end