function print_header_1()
    println("---------------------------------------------------------------------------------------")
    println("=======================================================================================")
    println("                  ProxSDP : Proximal Semidefinite Programming Solver                   ")
    println("                         (c) Mario Souto and Joaquim D. Garcia, 2020                   ")
    println("                                                              v1.8.4                   ")
    println("---------------------------------------------------------------------------------------")
end

function print_parameters(opt::Options, conic_sets::ConicSets)
    println("    Solver parameters:")
    tol_str = "       tol_gap = $(opt.tol_gap) tol_feasibility = $(opt.tol_feasibility)\n"
    tol_str *= "       tol_primal = $(opt.tol_primal) tol_dual = $(opt.tol_dual)"
    if length(conic_sets.socone) >= 1
        tol_str *= " tol_soc = $(opt.tol_soc)"
    end
    if length(conic_sets.sdpcone) >= 1
        tol_str *= " tol_psd = $(opt.tol_psd)"
    end
    println(tol_str)

    println("       max_iter = $(opt.max_iter_local) time_limit = $(opt.time_limit)s")

    return nothing
end

eq_plural(val::Integer) = ifelse(val != 1, "ies", "y")
eqs(val::Integer) = val > 0 ? "$(val) linear equalit$(eq_plural(val))" : ""
ineqs(val::Integer) = val > 0 ? "$(val) linear inequalit$(eq_plural(val))" : ""
function print_constraints(aff::AffineSets)
    println("    Constraints:")
    println("       $(eqs(aff.p)) and $(eqs(aff.m))")
    return nothing
end

cone_plural(val::Integer) = ifelse(val != 1, "s", "")
function print_prob_data(conic_sets::ConicSets)
    soc_dict = Dict()
    for soc in conic_sets.socone
        if soc.len in keys(soc_dict)
            soc_dict[soc.len] += 1
        else
            soc_dict[soc.len] = 1
        end
    end
    psd_dict = Dict()
    for psd in conic_sets.sdpcone
        if psd.sq_side in keys(psd_dict)
            psd_dict[psd.sq_side] += 1
        else
            psd_dict[psd.sq_side] = 1
        end
    end 
    println("    Cones:")
    if length(conic_sets.socone) > 0
        for (k, v) in soc_dict
            println("       $v second order cone$(cone_plural(v)) of size $k")
        end
    end
    if length(conic_sets.sdpcone) > 0
        for (k, v) in psd_dict
            println("       $v psd cone$(cone_plural(v)) of size $k")
        end
    end

    return nothing
end

function print_header_2(opt, beg = true)
    bar  = "---------------------------------------------------------------------------------------"
    name = "    Initializing Primal-Dual Hybrid Gradient method"
    cols = "|  iter  | prim obj | rel. gap |  feasb.  | prim res | dual res | tg. rank |  time(s) |"

    if opt.extended_log || opt.extended_log2
        bar  *= "-----------"
        cols *= " dual obj |"
    end
    if opt.extended_log2
        bar  *= "-----------"
        cols *= " d feasb. |"
    end

    if beg
        println(bar)
        println(name)
        println(bar)
    end
    println(cols)
    if beg
        println(bar)
    end

    return nothing
end

function print_progress(residuals::Residuals, p::Params, opt, val = -1.0)
    primal_res = residuals.primal_residual[p.iter]
    dual_res = residuals.dual_residual[p.iter]
    s_k = Printf.@sprintf("%d", p.iter)
    s_k *= " |"
    s_s = Printf.@sprintf("%.2e", residuals.dual_gap[p.iter])
    s_s *= " |"
    s_o = Printf.@sprintf("%.2e", residuals.prim_obj[p.iter])
    s_o *= " |"
    s_f = Printf.@sprintf("%.2e", residuals.feasibility[p.iter])
    s_f *= " |"
    s_p = Printf.@sprintf("%.2e", primal_res)
    s_p *= " |"
    s_d = Printf.@sprintf("%.2e", dual_res)
    s_d *= " |"
    s_target_rank = Printf.@sprintf("%g", sum(p.target_rank))
    s_target_rank *= " |"
    s_time = Printf.@sprintf("%g", time() - p.time0)
    s_time *= " |"
    a = "|"
    a *= " "^max(0, 9 - length(s_k))
    a *= s_k
    a *= " "^max(0, 11 - length(s_o))
    a *= s_o
    a *= " "^max(0, 11 - length(s_s))
    a *= s_s
    a *= " "^max(0, 11 - length(s_f))
    a *= s_f
    a *= " "^max(0, 11 - length(s_p))
    a *= s_p
    a *= " "^max(0, 11 - length(s_d))
    a *= s_d
    a *= " "^max(0, 11 - length(s_target_rank))
    a *= s_target_rank
    a *= " "^max(0, 11 - length(s_time))
    a *= s_time

    if opt.extended_log || opt.extended_log2
        str = Printf.@sprintf("%.3f", residuals.dual_obj[p.iter]) * " |"
        str = " "^max(0, 11 - length(str)) * str
        a *= str
    end
    if opt.extended_log2
        str = Printf.@sprintf("%.5f", val) * " |"
        str = " "^max(0, 11 - length(str)) * str
        a *= str
    end
    if opt.log_repeat_header
        print_header_2(opt, false)
    end

    println(a)

    return nothing
end

function print_result(stop_reason::Int, time_::Float64, residuals::Residuals, max_rank::Int,  p::Params)
    println("---------------------------------------------------------------------------------------")
    println("    Solver status:")
    println("       "*p.stop_reason_string)
    println("       Time elapsed     = $(round(time_; digits = 2)) seconds")
    println("       Primal objective = $(round(residuals.prim_obj[p.iter]; digits = 5))")
    println("       Dual objective   = $(round(residuals.dual_obj[p.iter]; digits = 5))")
    println("       Duality gap      = $(round(100*residuals.dual_gap[p.iter]; digits = 2)) %")
    println("---------------------------------------------------------------------------------------")
    println("    Primal feasibility:")
    println("       ||A(X) - b|| / (1 + ||b||) = $(round(residuals.equa_feasibility; digits = 6))    [linear equalities] ")
    println("       ||max(G(X) - h, 0)|| / (1 + ||h||) = $(round(residuals.ineq_feasibility; digits = 6))    [linear inequalities]")
    println("    Rank of p.s.d. variable is $max_rank.")
    println("=======================================================================================")
    
    return nothing
end
