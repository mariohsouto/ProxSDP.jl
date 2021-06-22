
function compute_gap!(residuals::Residuals, pair::PrimalDual, aff::AffineSets, p::Params)::Nothing

    # Duality gap
    residuals.prim_obj[p.iter] = dot(aff.c, pair.x)
    residuals.dual_obj[p.iter] = 0.
    if aff.p > 0
        residuals.dual_obj[p.iter] -= dot(aff.b, @view pair.y[1:aff.p])
    end
    if aff.m > 0
        residuals.dual_obj[p.iter] -= dot(aff.h, @view pair.y[aff.p+1:end])
    end
    residuals.duality_gap[p.iter] =
        abs(residuals.prim_obj[p.iter] - residuals.dual_obj[p.iter]) /
        (1. + abs(residuals.prim_obj[p.iter]) + abs(residuals.dual_obj[p.iter]))

    return nothing
end

function compute_primal_feasibility!(residuals::Residuals, a::AuxiliaryData, aff::AffineSets, p::Params, opt)::Nothing

    # Inplace primal feasibility residual
    if aff.p > 0
        residuals.equa_feasibility = 0.

        @simd for i in 1:aff.p     
            if opt.optimality_norm == "L2"  
                @inbounds residuals.equa_feasibility += (a.Mx[i] - aff.b[i])^2
            elseif opt.optimality_norm == "L_INF"
                @inbounds residuals.equa_feasibility = max(residuals.equa_feasibility, abs(a.Mx[i] - aff.b[i]))
            else
                error("Unknown optimality_norm")
            end
        end

    end
    if aff.m > 0
        residuals.ineq_feasibility = 0.

        @simd for i in aff.p+1:aff.p+aff.m
            if opt.optimality_norm == "L2"
                @inbounds residuals.ineq_feasibility += (max(0.0, a.Mx[i] - aff.h[i-aff.p]))^2
            elseif opt.optimality_norm == "L_INF"
                @inbounds residuals.ineq_feasibility = max(residuals.ineq_feasibility, a.Mx[i] - aff.h[i-aff.p])
            else
                error("Unknown optimality_norm")
            end
        end

    end

    # Save primal feasibility residual
    if opt.optimality_norm == "L2" 
        residuals.primal_feasibility[p.iter] = sqrt(residuals.equa_feasibility + residuals.ineq_feasibility) / (1. + p.norm_h + p.norm_b)
    elseif opt.optimality_norm == "L_INF"
        residuals.primal_feasibility[p.iter] = max(residuals.equa_feasibility, residuals.ineq_feasibility) / (1. + p.norm_h + p.norm_b)
    else
        error("Unknown optimality_norm")
    end

    return nothing
end

function compute_dual_feasibility!(residuals::Residuals, pair::PrimalDual, cones::ConicSets, a::AuxiliaryData, aff::AffineSets, p::Params, A, G, c, opt)::Nothing

    # Get dual constraints and cones
    dual_eq = pair.y[1:aff.p]
    dual_in = pair.y[aff.p+1:end]

    dual_cone = + c + A' * dual_eq + G' * dual_in 
    fix_diag_scaling(dual_cone, cones, 2.0)

    # Compute dual inequality feasibility
    ineq_viol = 0.0
    if length(dual_in) > 0
        if opt.optimality_norm == "L2"
            ineq_viol = sum(min.(0.0, dual_in).^2)
        elseif opt.optimality_norm == "L_INF"
            ineq_viol = -min(0.0, minimum(dual_in))
        else
            error("Unknown optimality_norm")
        end
    end

    # Compute dual cone feasibility
    cone_viol, cont = cone_feas(dual_cone, cones, a)
    zero_viol = 0.0
    dual_zr = dual_cone[(cont+1):end]
    if length(dual_zr) > 0
        if opt.optimality_norm == "L2"
            zero_viol = sum(dual_zr.^2)
        elseif opt.optimality_norm == "L_INF"
            zero_viol = maximum(abs.(dual_zr))
        else
            error("Unknown optimality_norm")
        end
    end

    # Save dual feasibility residual
    if opt.optimality_norm == "L2" 
        residuals.dual_feasibility[p.iter] = sqrt(cone_viol + ineq_viol + zero_viol) /  (1. + p.norm_c)
    elseif opt.optimality_norm == "L_INF"
        residuals.dual_feasibility[p.iter] = max(cone_viol, ineq_viol, zero_viol) /  (1. + p.norm_c)
    else
        error("Unknown optimality_norm")
    end

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

function cone_feas(v, cones, a, num = sqrt(2))
    sdp_viol = 0.0
    sdplen = psd_vec_to_square(v, a, cones, num) - 1
    for (idx, sdp) in enumerate(cones.sdpcone)
        if sdp.sq_side == 1
            sdp_viol = max(sdp_viol, -min(0.0, a.m[idx][1]))
        else
            fact = eigen!(a.m[idx])
            sdp_viol = max(sdp_viol, -min(0.0, minimum(fact.values)))
        end
    end
    soc_viol = 0.0
    cont = sdplen
    for (idx, soc) in enumerate(cones.socone)
        len = soc.len
        push!(a.soc_s, view(v, cont + 1))
        s = v[cont+1]
        sdp_viol = max(sdp_viol, -min(0.0, s - norm(view(v, cont + 2:cont + len))))
        cont += len
    end
    return max(sdp_viol, soc_viol), cont
end