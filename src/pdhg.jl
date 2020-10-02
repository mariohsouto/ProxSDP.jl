function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, opt)::CPResult#,
    # warm::WarmStart)::CPResult

    # Initialize parameters
    p = Params()
    p.theta = opt.initial_theta
    p.adapt_level = opt.initial_adapt_level
    p.window = opt.convergence_window
    p.beta = opt.initial_beta
    p.time0 = time()
    p.norm_b = norm(affine_sets.b, 2)
    p.norm_h = norm(affine_sets.h, 2)
    p.norm_c = norm(affine_sets.c, 2)
    p.rank_update, p.stop_reason, p.update_cont = 0, 0, 0
    p.stop_reason_string = "Not optimized"
    p.target_rank = 2 * ones(length(conic_sets.sdpcone))
    p.current_rank = 2 * ones(length(conic_sets.sdpcone))
    p.min_eig = zeros(length(conic_sets.sdpcone))
    p.dual_feasibility = -1.0
    p.dual_feasibility_check = false
    p.certificate_search = false
    p.certificate_search_min_iter = 0
    p.certificate_found = false
    sol = Array{CPResult}(undef, 0)
    arc_list = [EigSolverAlloc(Float64, sdp.sq_side, opt) for (idx, sdp) in enumerate(conic_sets.sdpcone)]
    ada_count = 0

    if opt.max_iter <= 0
        if length(conic_sets.socone) > 0 || length(conic_sets.sdpcone) > 0
            opt.max_iter_local = opt.max_iter_conic
        else
            opt.max_iter_local = opt.max_iter_lp
        end
    else
        opt.max_iter_local = opt.max_iter
    end

    # Print header
    if opt.log_verbose
        print_header_1()
        print_parameters(opt, conic_sets)
        print_constraints(affine_sets)
        if length(conic_sets.socone) + length(conic_sets.sdpcone) > 0
            print_prob_data(conic_sets)
        end
        print_header_2(opt)
    end

    @timeit "Init" begin

        # Scale objective function
        @timeit "normscale alloc" begin
            c_orig, var_ordering = preprocess!(affine_sets, conic_sets)
            A_orig, b_orig = copy(affine_sets.A), copy(affine_sets.b)
            G_orig, h_orig = copy(affine_sets.G), copy(affine_sets.h)
            rhs_orig = vcat(b_orig, h_orig)
        end

        # Diagonal preconditioning
        @timeit "equilibrate" begin
            if opt.equilibration
                M = vcat(affine_sets.A, affine_sets.G) 
                UB = maximum(M)
                LB = minimum(M)
                if LB/UB <= opt.equilibration_limit
                    opt.equilibration = false
                end
            end
            if opt.equilibration_force
                opt.equilibration = true
            end
            if opt.equilibration
                @timeit "equilibrate inner" E, D = equilibrate!(M, affine_sets, opt)
                @timeit "equilibrate scaling" begin
                    M = E * M * D
                    affine_sets.A = M[1:affine_sets.p, :]
                    affine_sets.G = M[affine_sets.p + 1:end, :]
                    rhs = E * rhs_orig
                    affine_sets.b = rhs[1:affine_sets.p]
                    affine_sets.h = rhs[affine_sets.p + 1:end]
                    affine_sets.c = D * affine_sets.c
                end
            else
                E = Diagonal(zeros(affine_sets.m + affine_sets.p))
                D = Diagonal(zeros(affine_sets.n))
            end
        end
        
        # Scale the off-diagonal entries associated with p.s.d. matrices by √2
        @timeit "normscale" norm_scaling(affine_sets, conic_sets)

        # Initialization
        pair = PrimalDual(affine_sets)
        a = AuxiliaryData(affine_sets, conic_sets)
        map_socs!(pair.x, conic_sets, a)
        residuals = Residuals(p.window)

        # Diagonal scaling
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'

        # Stepsize parameters and linesearch parameters
        if !opt.approx_norm
            @timeit "svd" if minimum(size(M)) >= 2
                try
                    spectral_norm = Arpack.svds(M, nsv=1)[1].S[1]
                catch
                    println("WARNING: Failed to compute spectral norm of M, shifting to Frobenius norm")
                    spectral_norm = norm(M)
                end
            else
                F = LinearAlgebra.svd!(Matrix(M))
                spectral_norm = maximum(F.S)
            end
        else
            spectral_norm = norm(M)
        end

        # Build struct for storing matrices
        mat = Matrices(M, Mt, affine_sets.c)

        # Initial primal and dual steps
        p.primal_step = 1. / spectral_norm
        p.primal_step_old = p.primal_step
        p.dual_step = p.primal_step

    end

    # Initialization
    if opt.advanced_initialization
        pair.x .= p.primal_step .* mat.c
        mul!(a.Mx, mat.M, pair.x)
        mul!(a.Mx_old, mat.M, pair.x_old)
    end

    # Fixed-point loop
    @timeit "CP loop" for k in 1:2*opt.max_iter_local

        # Update iterator
        p.iter = k

        # Primal step
        @timeit "primal" primal_step!(pair, a, conic_sets, mat, opt, p, arc_list, p.iter)

        # Linesearch (dual step)
        if opt.line_search_flag
            @timeit "linesearch" linesearch!(pair, a, affine_sets, mat, opt, p)
        else
            @timeit "dual step" dual_step!(pair, a, affine_sets, mat, opt, p)
        end

        # Compute residuals and update old iterates
        @timeit "residual" compute_residual!(residuals, pair, a, p, affine_sets)

        # Compute optimality gap and feasibility error
        @timeit "gap" compute_gap!(residuals, pair, a, affine_sets, p)

        if (opt.check_dual_feas && mod(k, opt.check_dual_feas_freq) == 0) ||
                (opt.log_verbose && mod(k, opt.log_freq) == 0 && opt.extended_log2)
            cc = ifelse(p.stop_reason == 6, 0.0, 1.0)*c_orig
            p.dual_feasibility = dual_feas(pair.y, conic_sets, affine_sets, cc, A_orig, G_orig, a)
            p.dual_feasibility_check = true
        else
            p.dual_feasibility_check = false
        end

        # Print progress
        if opt.log_verbose && mod(k, opt.log_freq) == 0
            print_progress(residuals, p, opt, p.dual_feasibility)
            # @show val
            # # stable feasibility and gap
            # @show max_abs_diff(residuals.feasibility)
            # @show max_abs_diff(residuals.dual_gap)
            # @show min_abs_diff(residuals.dual_gap)
            # # growing obj
            # @show min_abs_diff(residuals.prim_obj)
            # @show min_abs_diff(residuals.dual_obj)
        end

        if p.iter < p.certificate_search_min_iter
            continue
        end

        if opt.certificate_search && p.certificate_search
            if p.stop_reason == 6 # infeasible
                # dual ray
                # @show residuals.dual_obj[k]
                if residuals.dual_obj[k] > +opt.certificate_obj_tol
                    p.dual_feasibility = dual_feas(pair.y, conic_sets, affine_sets, 0*c_orig, A_orig, G_orig, a)
                    p.dual_feasibility_check = true
                    # @show tol
                    if p.dual_feasibility < opt.tol_feasibility_dual
                        p.certificate_found = true
                        if opt.log_verbose
                            println("Dual ray found.")
                        end
                        # println("\n\n\nGood for infeas\n\n\n")
                        break
                    end
                end
            else p.stop_reason == 5 # unbounded
                # primal ray
                # @show residuals.prim_obj[k]
                if residuals.prim_obj[k] < -opt.certificate_obj_tol
                    # @show tol = residuals.feasibility[p.iter]
                    if residuals.feasibility[p.iter] < opt.tol_feasibility
                        p.certificate_found = true
                        if opt.log_verbose
                        println("Primal ray found")
                        end
                        # println("\n\n\nGood for unbounded\n\n\n")
                        break
                    end
                end
            end
            if (residuals.prim_obj[k] < -opt.certificate_fail_tol &&
                    residuals.dual_obj[k] < -opt.certificate_fail_tol &&
                    residuals.feasibility[p.iter] < -opt.certificate_fail_tol
                    ) ||
                    isnan(residuals.comb_residual[k])
                if opt.log_verbose
                    println("Failed to finds certificate.")
                end
                p.stop_reason_string *= " [Failed to find certificate]"
                break
            end
        end

        # Check convergence
        p.rank_update += 1
        if residuals.dual_gap[p.iter] <= opt.tol_gap && residuals.feasibility[p.iter] <= opt.tol_feasibility &&
                (!opt.check_dual_feas || p.dual_feasibility < opt.tol_feasibility_dual)
            if convergedrank(a, p, conic_sets, opt) && soc_convergence(a, conic_sets, pair, opt, p) && p.iter > opt.min_iter
                if !p.certificate_search
                    p.stop_reason = 1 # Optimal
                    p.stop_reason_string = "Optimal solution found"
                else
                    # error("Certificate problem")
                    println("Failed to finds certificate - type 2.")
                    p.stop_reason_string *= " [Failed to find certificate - type 2]"
                    break
                end
                break
            elseif p.rank_update > p.window
                p.update_cont += 1
                if p.update_cont > 0
                    for (idx, sdp) in enumerate(conic_sets.sdpcone)
                        if p.current_rank[idx] + opt.rank_slack >= p.target_rank[idx]
                            if min_eig(a, idx, p) > opt.tol_psd
                                if opt.rank_increment == 0
                                    p.target_rank[idx] = min(opt.rank_increment_factor * p.target_rank[idx], sdp.sq_side)
                                else
                                    p.target_rank[idx] = min(opt.rank_increment_factor + p.target_rank[idx], sdp.sq_side)
                                end
                            end
                        end
                    end
                    p.rank_update, p.update_cont = 0, 0
                end
            end
        # Check divergence
        elseif k > p.window && residuals.comb_residual[k - p.window] < residuals.comb_residual[k] && p.rank_update > p.window
            p.update_cont += 1
            if p.update_cont > opt.divergence_min_update
                full_rank_flag = true
                for (idx, sdp) in enumerate(conic_sets.sdpcone)
                    if p.target_rank[idx] < sdp.sq_side
                        full_rank_flag = false
                        p.rank_update, p.update_cont = 0, 0
                    end
                    if p.current_rank[idx] + opt.rank_slack >= p.target_rank[idx]
                        if min_eig(a, idx, p) > opt.tol_psd
                            if opt.rank_increment == 0
                                p.target_rank[idx] = min(opt.rank_increment_factor * p.target_rank[idx], sdp.sq_side)
                            else
                                p.target_rank[idx] = min(opt.rank_increment_factor + p.target_rank[idx], sdp.sq_side)
                            end
                        end
                    end
                end
            end
        # Adaptive stepsizes
        elseif residuals.primal_residual[k] > opt.tol_primal && residuals.dual_residual[k] < opt.tol_dual && k > p.window
            ada_count += 1
            if ada_count > opt.adapt_window
                ada_count = 0
                if opt.line_search_flag
                    p.beta *= (1. - p.adapt_level)
                    p.primal_step /= sqrt(1. - p.adapt_level)
                else
                    p.primal_step /= (1. - p.adapt_level)
                    p.dual_step *= (1. - p.adapt_level)
                end
                p.adapt_level *= opt.adapt_decay
            end
        elseif residuals.primal_residual[k] < opt.tol_primal && residuals.dual_residual[k] > opt.tol_dual && k > p.window
            ada_count += 1
            if ada_count > opt.adapt_window
                ada_count = 0
                if opt.line_search_flag
                    p.beta /= (1. - p.adapt_level)
                    p.primal_step *= sqrt(1. - p.adapt_level)
                else
                    p.primal_step *= (1. - p.adapt_level)
                    p.dual_step /= (1. - p.adapt_level)
                end
                p.adapt_level *= opt.adapt_decay
            end
        end

        # max_iter or time limit stop condition
        if p.iter >= opt.max_iter_local || time() - p.time0 > opt.time_limit
            if p.iter > opt.min_iter_time_infeas &&
                max_abs_diff(residuals.dual_gap) < opt.infeas_stable_gap_tol &&
                residuals.dual_gap[k] > opt.infeas_limit_gap_tol # low gap but far from zero, say 10%
                if residuals.feasibility[p.iter] <= opt.tol_feasibility/100
                    p.stop_reason = 5 # Unbounded
                    p.stop_reason_string = "Problem declared unbounded due to lack of improvement"
                    if opt.certificate_search && !p.certificate_search
                        certificate_dual_infeasibility(affine_sets, p, opt)
                        push!(sol, cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
                            c_orig, A_orig, b_orig, G_orig, h_orig, D, E, var_ordering, a))
                        # println("6")
                    elseif opt.certificate_search && p.certificate_search
                        # error("6")
                    else
                        break
                    end
                elseif residuals.feasibility[p.iter] > opt.infeas_feasibility_tol
                    p.stop_reason = 6 # Infeasible
                    p.stop_reason_string = "Problem declared infeasible due to lack of improvement"
                    if opt.certificate_search && !p.certificate_search
                        certificate_infeasibility(affine_sets, p, opt)
                        push!(sol, cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
                            c_orig, A_orig, b_orig, G_orig, h_orig, D, E, var_ordering, a))
                        # println("7")
                    elseif opt.certificate_search && p.certificate_search
                        # error("7")
                    else
                        break
                    end
                end
            elseif p.iter >= opt.max_iter_local
                p.stop_reason = 3 # Iteration limit
                p.stop_reason_string = "Iteration limit of $(opt.max_iter_local) was hit"
                if opt.warn_on_limit
                    @warn("WARNING: Iteration limit hit.")
                end
            else
                p.stop_reason = 2 # Time limit
                p.stop_reason_string = "Time limit hit, limit: $(opt.time_limit) time: $(time() - p.time0)"
                if opt.warn_on_limit
                    println("WARNING: Time limit hit.")
                end
            end
            if p.iter >= opt.max_iter_local || time() - p.time0 > opt.time_limit
                break
            end
        end

        # already looking of certificates
        if opt.certificate_search && p.certificate_search
            continue
        end

        # Dual obj growing too much
        if (p.iter > opt.min_iter_max_obj && residuals.dual_obj[k] >  opt.max_obj) || isnan(residuals.dual_obj[k])
            # dual unbounded
            p.stop_reason = 6 # Infeasible
            p.stop_reason_string = "Infeasible: |Dual objective| = $(residuals.dual_obj[k]) > maximum allowed = $(opt.max_obj)"
            if opt.certificate_search && !p.certificate_search
                certificate_infeasibility(affine_sets, p, opt)
                push!(sol, cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
                    c_orig, A_orig, b_orig, G_orig, h_orig, D, E, var_ordering, a))
            else
                break
            end
        end
        # Primal obj growing too much
        if (p.iter > opt.min_iter_max_obj && residuals.prim_obj[k] < -opt.max_obj) || isnan(residuals.prim_obj[k])
            p.stop_reason = 5 # Unbounded
            p.stop_reason_string = "Unbounded: |Primal objective| = $(residuals.prim_obj[k]) > maximum allowed = $(opt.max_obj)"
            if opt.certificate_search && !p.certificate_search
                certificate_dual_infeasibility(affine_sets, p, opt)
                push!(sol, cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
                    c_orig, A_orig, b_orig, G_orig, h_orig, D, E, var_ordering, a))
            else
                break
            end
        end
        # stalled feasibility with meaningful gap
        if p.iter > opt.min_iter_max_obj &&
                residuals.dual_gap[k] > opt.infeas_limit_gap_tol && # low gap but far from zero, say 10% 
                residuals.feasibility[p.iter] > opt.infeas_feasibility_tol &&
                max_abs_diff(residuals.feasibility) < opt.infeas_stable_feasibility_tol
            
            p.stop_reason = 6 # Infeasible
            p.stop_reason_string = "Infeasible: feasibility stalled at $(residuals.feasibility[p.iter])"
            if opt.certificate_search && !p.certificate_search
                certificate_infeasibility(affine_sets, p, opt)
                push!(sol, cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
                    c_orig, A_orig, b_orig, G_orig, h_orig, D, E, var_ordering, a))
            else
                break
            end
        end
        # stalled gap at 100%
        if p.iter > opt.min_iter_max_obj &&
                residuals.dual_gap[k] > 1-opt.infeas_gap_tol &&
                max_abs_diff(residuals.dual_gap) < opt.infeas_stable_gap_tol
            if abs(residuals.dual_obj[k]) > abs(residuals.prim_obj[k]) && residuals.feasibility[p.iter] > opt.infeas_feasibility_tol
                # dual unbounded
                p.stop_reason = 6 # Infeasible
                p.stop_reason_string = "Infeasible: duality gap stalled at 100 % with |Dual objective| >> |Primal objective|"
                if opt.certificate_search && !p.certificate_search
                    certificate_infeasibility(affine_sets, p, opt)
                    push!(sol, cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
                        c_orig, A_orig, b_orig, G_orig, h_orig, D, E, var_ordering, a))
                else
                    break
                end
            elseif abs(residuals.prim_obj[k]) > abs(residuals.dual_obj[k]) && residuals.feasibility[p.iter] <= opt.tol_feasibility
                p.stop_reason = 5 # Unbounded
                p.stop_reason_string = "Unbounded: duality gap stalled at 100 % with |Dual objective| << |Primal objective|"
                if opt.certificate_search && !p.certificate_search
                    certificate_dual_infeasibility(affine_sets, p, opt)
                    push!(sol, cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
                        c_orig, A_orig, b_orig, G_orig, h_orig, D, E, var_ordering, a))
                else
                    break
                end
            end
        end
    end

    # Compute results
    time_ = time() - p.time0

    # Print result
    if opt.log_verbose
        val = -1.0
        if opt.extended_log2
            cc = ifelse(p.stop_reason == 6, 0.0, 1.0)*c_orig
            val = dual_feas(pair.y, conic_sets, affine_sets, cc, A_orig, G_orig, a)
        end
        print_progress(residuals, p, opt, val)
        print_result(
            p.stop_reason,
            time_,
            residuals,
            length(p.current_rank) > 0 ? maximum(p.current_rank) : 0,
            p)
    end

    if opt.certificate_search && p.certificate_search
        @assert length(sol) == 1
        if p.certificate_found
            if p.stop_reason == 6
                c_orig .*= 0.0
            end
            pop!(sol)
            push!(sol, cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
                c_orig, A_orig, b_orig, G_orig, h_orig, D, E, var_ordering, a))
        end
    else
        @assert length(sol) == 0
        push!(sol, cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
        c_orig, A_orig, b_orig, G_orig, h_orig, D, E, var_ordering, a))
    end

    return sol[1]
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, affine_sets::AffineSets, mat::Matrices, opt::Options, p::Params)::Nothing
    p.primal_step = p.primal_step * sqrt(1. + p.theta)

    for i in 1:opt.max_linsearch_steps
        p.theta = p.primal_step / p.primal_step_old

        @timeit "linesearch 1" begin
            a.y_half .= pair.y .+ (p.beta * p.primal_step) .* ((1. + p.theta) .* a.Mx .- p.theta .* a.Mx_old)
        end

        @timeit "linesearch 2" begin
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

        if sqrt(p.beta) * p.primal_step * Mty_norm <= opt.delta * y_norm
            break
        else
            p.primal_step *= opt.linsearch_decay
        end
    end

    # Reverte in-place norm
    a.Mty .+= a.Mty_old
    a.y_temp .+= pair.y_old

    copyto!(pair.y, a.y_temp)
    p.primal_step_old = p.primal_step
    p.dual_step = p.beta * p.primal_step

    return nothing
end

function dual_step!(pair::PrimalDual, a::AuxiliaryData, affine_sets::AffineSets, mat::Matrices, opt::Options, p::Params)::Nothing

    @timeit "dual step 1" begin
        a.y_half .= pair.y .+ p.dual_step * (2. * a.Mx .- a.Mx_old)
    end

    @timeit "dual step 2" begin
        copyto!(a.y_temp, a.y_half)
        box_projection!(a.y_half, affine_sets, p.dual_step)
        a.y_temp .-= p.dual_step * a.y_half
    end

    @timeit "linesearch 3" mul!(a.Mty, mat.Mt, a.y_temp)

    copyto!(pair.y, a.y_temp)
    p.primal_step_old = p.primal_step

    return nothing
end

function primal_step!(pair::PrimalDual, a::AuxiliaryData, cones::ConicSets, mat::Matrices, opt::Options, p::Params, arc_list, iter::Int64)::Nothing

    pair.x .-= p.primal_step .* (a.Mty .+ mat.c)

    # Projection onto the p.s.d. cone
    if length(cones.sdpcone) >= 1
        @timeit "sdp proj" psd_projection!(pair.x, a, cones, opt, p, arc_list, iter)
    end

    # Projection onto the second order cone
    if length(cones.socone) >= 1
        @timeit "soc proj" soc_projection!(pair.x, a, cones, opt, p)
    end

    @timeit "linesearch -1" mul!(a.Mx, mat.M, pair.x)

    return nothing
end

function certificate_dual_infeasibility(affine_sets, p, opt)
    if opt.log_verbose
        println("Begin search for dual infeasibility certificate")
    end
    fill!(affine_sets.b, 0.0)
    fill!(affine_sets.h, 0.0)
    certificate_parameters(p, opt)
    return nothing
end
function certificate_infeasibility(affine_sets, p, opt)
    if opt.log_verbose
        println("Begin search for infeasibility certificate")
    end
    fill!(affine_sets.c, 0.0)
    certificate_parameters(p, opt)
    return nothing
end
function certificate_parameters(p, opt)
    # println("\n\n\n")
    p.certificate_search_min_iter = p.iter + 2 * opt.convergence_window + div(p.iter, 5) + 1000
    p.certificate_search = true
    opt.time_limit *= 1.1
    opt.max_iter_local = opt.max_iter_local + div(opt.max_iter_local, 10)
    return nothing
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

function get_duals(y::Vector{T}, cones::ConicSets, affine::AffineSets, c, A, G) where T

    dual_eq = y[1:affine.p]
    dual_in = y[affine.p+1:end]

    dual_cone = + c + A' * dual_eq + G' * dual_in
    fix_diag_scaling(dual_cone, cones, 2.0)

    return dual_eq, dual_in, dual_cone
end

function dual_feas(y::Vector{T}, cones::ConicSets, affine::AffineSets, c, A, G, a) where T
    dual_eq, dual_in, dual_cone = get_duals(y, cones, affine, c, A, G)
    return dual_feas(dual_in, dual_cone, cones, a)
end
function dual_feas(dual_in::Vector{T}, dual_cone::Vector{T}, cones, a) where T

    ineq_viol = 0.0
    if length(dual_in) > 0
        # @show minimum(dual_in)
        # @show maximum(dual_in)
        ineq_viol = -min(0.0, minimum(dual_in))
    end

    cone_viol, cont = cone_feas(dual_cone, cones, a)

    zero_viol = 0.0
    dual_zr = dual_cone[(cont+1):end]
    if length(dual_zr) > 0
        zero_viol = maximum(abs.(dual_zr))
    end
    # @show cone_viol, ineq_viol, zero_viol
    return max(cone_viol, ineq_viol, zero_viol)
end

function fix_diag_scaling(v, cones, num)
    # Remove diag scaling
    cont = 1
    @inbounds for sdp in cones.sdpcone, j in 1:sdp.sq_side, i in 1:j#j:sdp.sq_side
        if i != j
            v[cont] /= num
        end
        cont += 1
    end
end

function cache_solution(pair, residuals, conic_sets, affine_sets, p, opt,
    c, A, b, G, h, D, E, var_ordering, a)

    # Remove diag scaling
    fix_diag_scaling(pair.x, conic_sets, sqrt(2.0))

    # Remove equilibrating
    if opt.equilibration
        pair.x = D * pair.x
        pair.y = E * pair.y
    end

    slack_eq   = A * pair.x - b
    slack_ineq = G * pair.x - h

    dual_eq, dual_in, dual_cone =
        get_duals(pair.y, conic_sets, affine_sets, c, A, G)

    dual_feasibility = dual_feas(dual_in, dual_cone, conic_sets, a)

    # Post processing
    # pair.x = pair.x[var_ordering]
    # dual_cone = dual_cone[var_ordering]

    return CPResult(
        p.stop_reason,
        p.stop_reason_string,
        pair.x[var_ordering],
        # pair.y,
        dual_cone[var_ordering],
        dual_eq,
        dual_in,
        slack_eq,
        slack_ineq,
        residuals.equa_feasibility,
        residuals.ineq_feasibility,
        residuals.prim_obj[p.iter],
        residuals.dual_obj[p.iter],
        residuals.dual_gap[p.iter],
        time() - p.time0,
        sum(p.current_rank),
        residuals.feasibility[p.iter] <= opt.tol_feasibility,
        dual_feasibility <= opt.tol_feasibility_dual,
        p.certificate_found,
    )
end