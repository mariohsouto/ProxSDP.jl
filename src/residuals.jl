
function compute_gap!(residuals::Residuals, pair::PrimalDual, a::AuxiliaryData, aff::AffineSets, p::Params)::Nothing

    # Inplace primal feasibility error
    if aff.p > 0
        residuals.equa_feasibility = 0.
        @simd for i in 1:aff.p
            @inbounds residuals.equa_feasibility = max(residuals.equa_feasibility, abs(a.Mx[i] - aff.b[i]))
        end
        residuals.equa_feasibility /= (1. + p.norm_b)
    end
    if aff.m > 0
        residuals.ineq_feasibility = 0.
        @simd for i in aff.p+1:aff.p+aff.m
            @inbounds residuals.ineq_feasibility = max(residuals.ineq_feasibility, max(a.Mx[i] - aff.h[i-aff.p], 0.0))
        end
        residuals.ineq_feasibility /= (1. + p.norm_h)
    end
    residuals.feasibility = max(residuals.equa_feasibility, residuals.ineq_feasibility)

    # Primal-dual gap
    residuals.prim_obj = dot(aff.c, pair.x)
    residuals.dual_obj = 0.
    if aff.p > 0
        residuals.dual_obj -= dot(aff.b, @view pair.y[1:aff.p])
    end
    if aff.m > 0
        residuals.dual_obj -= dot(aff.h, @view pair.y[aff.p+1:end])
    end
    residuals.dual_gap[p.iter] = abs(residuals.prim_obj - residuals.dual_obj) / (1. + abs(residuals.prim_obj) + abs(residuals.dual_obj))

    return nothing
end

function compute_residual!(residuals::Residuals, pair::PrimalDual, a::AuxiliaryData, p::Params, aff::AffineSets)::Nothing

    # Primal residual
    # Px_old
    a.Mty_old .= pair.x_old .- p.primal_step .* a.Mty_old
    # Px
    pair.x_old .= pair.x .- p.primal_step .* a.Mty
    # Px - Px_old
    pair.x_old .-= a.Mty_old 
    residuals.primal_residual[p.iter] = sqrt(aff.n) * norm(pair.x_old, Inf) / max(norm(a.Mty_old, Inf), p.norm_b, p.norm_h, 1.)

    # Dual residual
    # Py_old
    a.Mx_old .= pair.y_old .- p.dual_step .* a.Mx_old
    # Py
    pair.y_old .= pair.y .- p.dual_step .* a.Mx
    # Py - Py_old
    pair.y_old .-= a.Mx_old
    residuals.dual_residual[p.iter] = sqrt(aff.m + aff.p) * norm(pair.y_old, Inf) / max(norm(a.Mx_old, Inf), p.norm_c, 1.)

    # Compute combined residual
    residuals.comb_residual[p.iter] = max(residuals.primal_residual[p.iter], residuals.dual_residual[p.iter])

    # Keep track of previous iterates
    copyto!(pair.x_old, pair.x)
    copyto!(pair.y_old, pair.y)
    copyto!(a.Mty_old, a.Mty)
    copyto!(a.Mx_old, a.Mx)

    return nothing
end

function soc_convergence(a::AuxiliaryData, cones::ConicSets, pair::PrimalDual, opt::Options, p::Params)::Bool
    for (idx, soc) in enumerate(cones.socone)
        if soc_gap(a.soc_v[idx], a.soc_s[idx]) >= opt.tol_soc
            return false
        end
    end

    return true
end

function soc_gap(v::ViewVector, s::ViewScalar)

    return norm(v, 2) - s[]
end

function convergedrank(a, p::Params, cones::ConicSets, opt::Options)::Bool
    for (idx, sdp) in enumerate(cones.sdpcone)
        if !(
                sdp.sq_side < opt.min_size_krylov_eigs ||
                p.target_rank[idx] > opt.max_target_rank_krylov_eigs ||
                min_eig(a, idx, p) < opt.tol_psd
            )
            # @show min_eig(a, idx, p), -opt.tol_psd
            return false
        end
    end

    return true
end