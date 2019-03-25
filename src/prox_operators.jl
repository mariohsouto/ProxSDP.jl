
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
                        if unsafe_getvalues(arc[idx])[i] > opt.tol_psd
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
    fact = eigen!(a.m[idx])
    fill!(a.m[idx].data, 0.0)
    for i in 1:length(fact.values)
        if fact.values[i] > opt.tol_psd
            current_rank += 1
            LinearAlgebra.BLAS.gemm!('N', 'T', fact.values[i], fact.vectors[:, i], fact.vectors[:, i], 1.0, a.m[idx].data)
        end
    end
    return nothing
end

function so_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, cones::ConicSets, opt::Options, p::Params)
    for (idx, soc) in enumerate(cones.socone)
        soc_projection!(a.soc_v[idx], a.soc_s[idx])
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