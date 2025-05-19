
function preprocess!(aff::AffineSets, conic_sets::ConicSets)
    c_orig = zeros(1)
    if length(conic_sets.sdpcone) >= 1 || length(conic_sets.socone) >= 1
        all_cone_vars = Int[]
        for (idx, sdp) in enumerate(conic_sets.sdpcone)
            M = zeros(Int, sdp.sq_side, sdp.sq_side)
            append!(all_cone_vars, conic_sets.sdpcone[idx].vec_i)
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

function norm_scaling(affine_sets::AffineSets, cones::ConicSets)
    cte = (sqrt(2.) / 2.)
    rows = SparseArrays.rowvals(affine_sets.A)
    cont = 1
    for sdp in cones.sdpcone, j in 1:sdp.sq_side, i in 1:j
        if i != j
            for line in SparseArrays.nzrange(affine_sets.A, cont)
                affine_sets.A[rows[line], cont] *= cte
            end
        end
        cont += 1
    end
    rows = SparseArrays.rowvals(affine_sets.G)
    cont = 1
    for sdp in cones.sdpcone, j in 1:sdp.sq_side, i in 1:j
        if i != j
            for line in SparseArrays.nzrange(affine_sets.G, cont)
                affine_sets.G[rows[line], cont] *= cte
            end
        end
        cont += 1
    end
    cont = 1
    @inbounds for sdp in cones.sdpcone, j in 1:sdp.sq_side, i in 1:j
        if i != j
            affine_sets.c[cont] *= cte
        end
        cont += 1
    end
    return nothing
end
