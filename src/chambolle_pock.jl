
using Debugger

struct CPResult
    status::Int
    primal::Vector{Float64}
    dual::Vector{Float64}
    slack::Vector{Float64}
    primal_residual::Float64
    dual_residual::Float64
    objval::Float64
    dual_objval::Float64
    gap::Float64
    time::Float64
end

function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, opt)::CPResult

    p = Params()

    p.theta = opt.initial_theta             # Overrelaxation parameter
    p.adapt_level = opt.initial_adapt_level # Factor by which the stepsizes will be balanced 
    p.window = opt.convergence_window       # Convergence check window
    p.beta = opt.initial_beta

    p.time0 = time()

    p.norm_rhs = norm(vcat(affine_sets.b, affine_sets.h))
    p.norm_c = norm(affine_sets.c)

    p.rank_update, p.converged, p.update_cont = 0, false, 0
    p.target_rank = 2*ones(length(conic_sets.sdpcone))
    p.current_rank = 2*ones(length(conic_sets.sdpcone))
    p.min_eig = zeros(length(conic_sets.sdpcone))

    analysis = false
    if analysis
        p.window = 1
    end

    if opt.log_verbose
        printheader()
    end

    @timeit "Init" begin
        # Scale objective function
        c_orig, var_ordering = preprocess!(affine_sets, conic_sets)
        A_orig, b_orig = copy(affine_sets.A), copy(affine_sets.b)
        G_orig, h_orig = copy(affine_sets.G), copy(affine_sets.h)
        rhs_orig = vcat(affine_sets.b, affine_sets.h)
        norm_scaling(affine_sets, conic_sets)

        # Initialization
        pair = PrimalDual(affine_sets)
        a = AuxiliaryData(affine_sets, conic_sets)
        arc = [ARPACKAlloc(Float64, 1) for i in conic_sets.sdpcone]
        map_socs!(pair.x, conic_sets, a)
        primal_residual = CircularVector{Float64}(2*p.window)
        dual_residual = CircularVector{Float64}(2*p.window)
        comb_residual = CircularVector{Float64}(2*p.window)

        # Diagonal scaling
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'
        rhs = vcat(affine_sets.b, affine_sets.h)
        mat = Matrices(M, Mt, rhs, affine_sets.c)
        mul!(a.Mtrhs, mat.Mt, rhs)
        
        # Stepsize parameters and linesearch parameters
        if minimum(size(M)) >= 2
            p.primal_step = (1. - 1e-5) / Arpack.svds(M, nsv = 1)[1].S[1]
        else
            p.primal_step = (1. - 1e-5) / maximum(LinearAlgebra.svd(Matrix(M)).S)
        end
        p.primal_step_old = p.primal_step
        p.dual_step = p.primal_step

        analysis = false
        linesearch_flag = true
    end

    # Fixed-point loop
    @timeit "CP loop" for k in 1:opt.max_iter
        p.iter = k

        # Primal update
        @timeit "primal" primal_step!(pair, a, conic_sets, mat, arc, opt, p)

        # Dual update
        if linesearch_flag
            linesearch!(pair, a, affine_sets, mat, opt, p)
        else
            dual_step!(pair, a, affine_sets, mat, opt, p)
        end

        # Compute residuals and update old iterates
        @timeit "residual" compute_residual!(pair, a, primal_residual, dual_residual, comb_residual, mat, p, affine_sets)
        
        # Print progress
        if opt.log_verbose && mod(k, opt.log_freq) == 0
            @show p.beta
            print_progress(primal_residual[k], dual_residual[k], p)
        end

        # Check convergence of inexact fixed-point
        p.rank_update += 1
        if primal_residual[k] < opt.tol_primal && dual_residual[k] < opt.tol_dual && k > opt.convergence_check
            if convergedrank(p, conic_sets, opt) && soc_convergence(a, conic_sets, pair, opt, p)
                p.converged = true
                best_prim_residual, best_dual_residual = primal_residual[k], dual_residual[k]
                if opt.log_verbose
                    print_progress(primal_residual[k], dual_residual[k], p)
                end
                break
            elseif p.rank_update > p.window
                p.update_cont += 1
                if p.update_cont > 0
                    for (idx, sdp) in enumerate(conic_sets.sdpcone)
                        p.target_rank[idx] = min(2 * p.target_rank[idx], sdp.sq_side)
                    end
                    p.rank_update, p.update_cont = 0, 0
                end
            end

        # Check divergence
        elseif k > p.window && comb_residual[k - p.window] < 0.99 * comb_residual[k] && p.rank_update > p.window
            p.update_cont += 1
            if p.update_cont > 50
                for (idx, sdp) in enumerate(conic_sets.sdpcone)
                    p.target_rank[idx] = min(2 * p.target_rank[idx], sdp.sq_side)
                end
                p.rank_update, p.update_cont = 0, 0
            end

        # Adaptive stepsizes
        elseif primal_residual[k] > opt.tol_primal && dual_residual[k] < opt.tol_dual && k > p.window
            if linesearch_flag
                p.beta *= (1 - p.adapt_level)
                if p.beta <= opt.min_beta
                    p.beta = opt.min_beta
                else
                    p.adapt_level *= opt.adapt_decay
                end
                if analysis && opt.log_verbose
                    println("Debug: Beta = $(p.beta), AdaptLevel = $(p.adapt_level)")
                end
            else
                p.dual_step *= (1. - p.adapt_level)
                p.primal_step /= (1. - p.adapt_level)
                p.adapt_level *= opt.adapt_decay
                if analysis && opt.log_verbose
                    println("Step: (Primal, Dual) = $((p.primal_step, p.dual_step)), AdaptLevel = $(p.adapt_level)")
                end
            end
        elseif primal_residual[k] < opt.tol_primal && dual_residual[k] > opt.tol_dual && k > p.window
            if linesearch_flag
                p.beta /= (1 - p.adapt_level)
                if p.beta >= opt.max_beta
                    p.beta = opt.max_beta
                else
                    p.adapt_level *= opt.adapt_decay
                end
                if analysis && opt.log_verbose
                    println("Debug: Beta = $(p.beta), AdaptLevel = $(p.adapt_level)")
                end
            else
                p.dual_step /= (1. - p.adapt_level)
                p.primal_step *= (1. - p.adapt_level)
                p.adapt_level *= opt.adapt_decay
                if analysis && opt.log_verbose
                    println("Step: (Primal, Dual) = $((p.primal_step, p.dual_step)), AdaptLevel = $(p.adapt_level)")
                end
            end
        end

        # # Adaptive reduce rank (heuristics)
        # if p.rank_update > 3 * p.window && comb_residual[k - p.window] > comb_residual[k]
        #     for (idx, sdp) in enumerate(conic_sets.sdpcone)
        #         p.target_rank[idx] = min(p.target_rank[idx], sdp.sq_side, p.current_rank[idx] + 10)
        #     end
        #     p.rank_update = 0
        # end
    end

    cont = 1
    @inbounds for sdp in conic_sets.sdpcone, j in 1:sdp.sq_side, i in j:sdp.sq_side
        if i != j
            pair.x[cont] /= sqrt(2.0)
        end
        cont += 1
    end

    # Compute results
    time_ = time() - p.time0
    prim_obj = dot(c_orig, pair.x)
    dual_obj = - dot(rhs_orig, pair.y)
    slack = A_orig * pair.x - b_orig
    slack2 = G_orig * pair.x - h_orig
    res_eq = norm(slack) / (1 + norm(b_orig))
    res_dual = prim_obj - dual_obj
    gap = (prim_obj - dual_obj) / abs(prim_obj) * 100
    pair.x = pair.x[var_ordering]

    ctr_primal = Float64[]
    for soc in conic_sets.socone
        append!(ctr_primal, pair.x[soc.idx])
    end
    for sdp in conic_sets.sdpcone
        append!(ctr_primal, pair.x[sdp.vec_i])
    end

    if opt.log_verbose
        println("----------------------------------------------------------------------")
        if p.converged
            println(" Solution metrics [solved]:")
        else
            println(" Solution metrics [failed to converge]:")
        end
        println(" Primal objective = $(round(prim_obj; digits = 5))")
        println(" Dual objective = $(round(dual_obj; digits = 5))")
        println(" Duality gap (%) = $(round(gap; digits = 2)) %")
        println(" Duality residual = $(round(res_dual; digits = 5))")
        println(" ||A(X) - b|| / (1 + ||b||) = $(round(res_eq; digits = 6))")
        println(" time elapsed = $(round(time_; digits = 6))")
        println("======================================================================")
    end
    return CPResult(Int(p.converged), pair.x, pair.y, -vcat(slack, slack2, -ctr_primal), res_eq, res_dual, prim_obj, dual_obj, gap, time_)
end

function dual_step!(pair::PrimalDual, a::AuxiliaryData, affine_sets::AffineSets, mat::Matrices, opt::Options, p::Params)

    a.y_half .= pair.y .+ p.dual_step .* ((1. + p.theta) .* a.Mx .- p.theta .* a.Mx_old)
    copyto!(a.y_temp, a.y_half)
    box_projection!(a.y_half, affine_sets, p.dual_step)
    a.y_temp .-= p.dual_step .* a.y_half

    @timeit "Mt * y" mul!(a.Mty, mat.Mt, a.y_temp)
        
    copyto!(pair.y, a.y_temp)

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

    @timeit "M * x" mul!(a.Mx, mat.M, pair.x)

    return nothing
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, affine_sets::AffineSets, mat::Matrices, opt::Options, p::Params)
    delta = 1. - 1e-3
    cont = 0
    p.primal_step_old = p.primal_step
    p.primal_step = p.primal_step * sqrt(1.0 + p.theta)
    for i in 1:opt.max_linsearch_steps
        cont += 1
        p.theta = p.primal_step / p.primal_step_old

        @timeit "linesearch 1" begin
            a.y_half .= pair.y .+ (p.beta * p.primal_step) .* ((1.0 + p.theta) .* a.Mx .- p.theta .* a.Mx_old)
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
    p.dual_step = p.beta * p.primal_step
    
    return nothing
end
