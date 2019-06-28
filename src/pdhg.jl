function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, opt)::CPResult

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
    p.target_rank = 2 * ones(length(conic_sets.sdpcone))
    p.current_rank = 2 * ones(length(conic_sets.sdpcone))
    p.min_eig = zeros(length(conic_sets.sdpcone))

    # Print header
    if opt.log_verbose
        print_header_1()
        print_parameters(opt, conic_sets)
        print_constraints(affine_sets)
        if length(conic_sets.socone) + length(conic_sets.sdpcone) > 0
            print_prob_data(conic_sets)
        end
        print_header_2()
    end

    @timeit "Init" begin

        # Scale objective function
        c_orig, var_ordering = preprocess!(affine_sets, conic_sets)
        A_orig, b_orig = copy(affine_sets.A), copy(affine_sets.b)
        G_orig, h_orig = copy(affine_sets.G), copy(affine_sets.h)
        rhs_orig = vcat(b_orig, h_orig)

        @show spectral_norm = Arpack.svds(affine_sets.A, nsv = 1)[1].S[1] 
        
        # Scale the off-diagonal entries associated with p.s.d. matrices by √2
        norm_scaling(affine_sets, conic_sets)

        # Diagonal preconditioning
        equilibrate = true
        if equilibrate
            E, Einv, D, Dinv = equilibrate!(affine_sets.A, affine_sets.A', affine_sets)
            affine_sets.A = E * affine_sets.A * D
            affine_sets.b = E * affine_sets.b
            affine_sets.c = D * affine_sets.c
        end

        # Initialization
        pair = PrimalDual(affine_sets)
        a = AuxiliaryData(affine_sets, conic_sets)
        map_socs!(pair.x, conic_sets, a)
        residuals = Residuals(p.window)

        # Diagonal scaling
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'

        # Stepsize parameters and linesearch parameters
        if minimum(size(M)) >= 2
            @show spectral_norm = Arpack.svds(M, nsv = 1)[1].S[1] 
        else
            spectral_norm = maximum(LinearAlgebra.svd(M).S)
        end

        # Normalize the linear system by the spectral norm of M
        spectral_norm_scaling = false
        if spectral_norm_scaling
            M /= spectral_norm
            Mt /= spectral_norm
            affine_sets.b /= sqrt(spectral_norm)
            affine_sets.h /= sqrt(spectral_norm)
            affine_sets.c /= sqrt(spectral_norm)
            p.primal_step = 1.
        else
            p.primal_step = 1. / spectral_norm
        end

        # Build struct for storing matrices
        mat = Matrices(M, Mt, affine_sets.c)

        # Initial primal and dual steps
        p.primal_step_old = p.primal_step
        p.dual_step = p.primal_step
    end

    # Fixed-point loop
    @timeit "CP loop" for k in 1:opt.max_iter

        # Update iterator
        p.iter = k

        # Primal step
        @timeit "primal" primal_step!(pair, a, conic_sets, mat, opt, p)

        # Linesearch (dual step)
        @timeit "linesearch" linesearch!(pair, a, affine_sets, mat, opt, p)

        # Compute residuals and update old iterates
        @timeit "residual" compute_residual!(residuals, pair, a, p, affine_sets)

        # Compute optimality gap and feasibility error
        @timeit "gap" compute_gap!(residuals, pair, a, affine_sets, p, E, D, A_orig, b_orig, c_orig)

        # Print progress
        if opt.log_verbose && mod(k, opt.log_freq) == 0
            print_progress(residuals, p)
        end
      
        # Check convergence
        p.rank_update += 1
        if residuals.dual_gap <= opt.tol_primal && residuals.equa_feasibility <= opt.tol_primal
            if convergedrank(p, conic_sets, opt) && soc_convergence(a, conic_sets, pair, opt, p)
                p.stop_reason = 1 # Optimal
                if opt.log_verbose
                    print_progress(residuals, p)
                end

                break

            elseif p.rank_update > p.window
                p.update_cont += 1
                if p.update_cont > 0
                    for (idx, sdp) in enumerate(conic_sets.sdpcone)
                        if p.current_rank[idx] + opt.rank_slack >= p.target_rank[idx]
                            if p.min_eig[idx] > opt.tol_psd
                                p.target_rank[idx] = min(2 * p.target_rank[idx], sdp.sq_side)
                            end
                        end
                    end
                    p.rank_update, p.update_cont = 0, 0
                end
            end

        # Check divergence
        elseif k > p.window && residuals.comb_residual[k - p.window] < residuals.comb_residual[k] && p.rank_update > p.window
            p.update_cont += 1
            if p.update_cont > 50
                for (idx, sdp) in enumerate(conic_sets.sdpcone)
                    if p.current_rank[idx] + opt.rank_slack >= p.target_rank[idx]
                        if p.min_eig[idx] > opt.tol_psd
                            p.target_rank[idx] = min(2 * p.target_rank[idx], sdp.sq_side)
                        end
                    end
                end
                p.rank_update, p.update_cont = 0, 0
            end
        
        # Adaptive stepsizes
        elseif residuals.primal_residual[k] > opt.tol_primal && residuals.dual_residual[k] < opt.tol_dual && k > p.window
            p.beta *= (1. - p.adapt_level)
            if p.beta <= opt.min_beta
                p.beta = opt.min_beta
            else
                p.adapt_level *= opt.adapt_decay
            end
            
        elseif residuals.primal_residual[k] < opt.tol_primal && residuals.dual_residual[k] > opt.tol_dual && k > p.window
            p.beta /= (1. - p.adapt_level)
            if p.beta >= opt.max_beta
                p.beta = opt.max_beta
            else
                p.adapt_level *= opt.adapt_decay
            end
        end

        # time_limit stop condition
        if time() - p.time0 > opt.time_limit
            if opt.log_verbose
                print_progress(residuals, p)
            end
            p.stop_reason = 2 # Time limit
            break
        end

        # max_iter stop condition
        if opt.max_iter == p.iter
            if opt.log_verbose
                print_progress(residuals, p)
            end
            p.stop_reason = 3 # Iteration limit
        end
    end

    @show prim_obj = dot(c_orig, pair.x)
    @show dual_obj = - dot(rhs_orig, pair.y)

    if equilibrate
        pair.x = D * pair.x
        pair.y = E * pair.y
    end

    @show prim_obj = dot(c_orig, pair.x)
    @show dual_obj = - dot(rhs_orig, pair.y)

    # Remove scaling
    cont = 1
    @inbounds for sdp in conic_sets.sdpcone, j in 1:sdp.sq_side, i in j:sdp.sq_side
        if i != j
            pair.x[cont] /= sqrt(2.)
        end
        cont += 1
    end
    if spectral_norm_scaling
        pair.x ./= sqrt(spectral_norm)
        pair.y ./= sqrt(spectral_norm)
        M *= spectral_norm
        Mt *= spectral_norm
    end

    @show prim_obj = dot(c_orig, pair.x)
    @show dual_obj = - dot(rhs_orig, pair.y)

    # Compute results
    time_ = time() - p.time0

    # Print result
    if opt.log_verbose
        print_result(p.stop_reason, time_, residuals, maximum(p.current_rank))
    end

    # Post processing
    pair.x = pair.x[var_ordering]
    ctr_primal = Float64[]
    for soc in conic_sets.socone
        append!(ctr_primal, pair.x[soc.idx])
    end
    for sdp in conic_sets.sdpcone
        append!(ctr_primal, pair.x[sdp.vec_i])
    end

    return CPResult(p.stop_reason, pair.x, pair.y, -vcat(a.Mx[1:affine_sets.p], a.Mx[affine_sets.p+1:end], -ctr_primal), residuals.equa_feasibility, residuals.ineq_feasibility, residuals.prim_obj, residuals.dual_obj, residuals.dual_gap, time_)
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, affine_sets::AffineSets, mat::Matrices, opt::Options, p::Params)
    cont = 0
    p.primal_step = p.primal_step * sqrt(1. + p.theta)
    
    for i in 1:opt.max_linsearch_steps
        cont += 1
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

    return nothing
end

function primal_step!(pair::PrimalDual, a::AuxiliaryData, cones::ConicSets, mat::Matrices, opt::Options, p::Params)

    pair.x .-= p.primal_step .* (a.Mty .+ mat.c)

    # Projection onto the p.s.d. cone
    if length(cones.sdpcone) >= 1
        @timeit "sdp proj" psd_projection!(pair.x, a, cones, opt, p)
    end

    # Projection onto the second order cone
    if length(cones.socone) >= 1
        @timeit "soc proj" soc_projection!(pair.x, a, cones, opt, p)
    end

    @timeit "linesearch -1" mul!(a.Mx, mat.M, pair.x)

    return nothing
end