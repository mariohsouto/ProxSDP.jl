Base.@kwdef mutable struct Options

    # Printing options
    log_verbose::Bool = false
    log_freq::Int = 1000
    timer_verbose::Bool = false
    timer_file::Bool = false
    disable_julia_logger::Bool = true

    # time options
    time_limit::Float64 = 3600_00. #100 hours

    warn_on_limit::Bool = false
    extended_log::Bool = false
    extended_log2::Bool = false
    log_repeat_header::Bool = false

    # Default tolerances
    tol_gap::Float64 = 1e-4
    tol_feasibility::Float64 = 1e-4
    tol_feasibility_dual::Float64 = 1e-4
    tol_primal::Float64 = 1e-4
    tol_dual::Float64 = 1e-4
    tol_psd::Float64 = 1e-7
    tol_soc::Float64 = 1e-7

    check_dual_feas::Bool = false
    check_dual_feas_freq::Int = 1000

    max_obj::Float64 = 1e20
    min_iter_max_obj::Int = 10

    # infeasibility check
    min_iter_time_infeas::Int = 1000
    infeas_gap_tol::Float64 = 1e-4
    infeas_limit_gap_tol::Float64 = 1e-1
    infeas_stable_gap_tol::Float64 = 1e-4
    infeas_feasibility_tol::Float64 = 1e-4
    infeas_stable_feasibility_tol::Float64 = 1e-8

    certificate_search::Bool = true
    certificate_obj_tol::Float64 = 1e-1
    certificate_fail_tol::Float64 = 1e-8

    # Bounds on beta (dual_step / primal_step) [larger bounds may lead to numerical inaccuracy]
    min_beta::Float64 = 1e-5
    max_beta::Float64 = 1e+5
    initial_beta::Float64 = 1.

    # Adaptive primal-dual steps parameters [adapt_decay above .7 may lead to slower convergence]
    initial_adapt_level::Float64 = .9
    adapt_decay::Float64 = .8
    adapt_window::Int64 = 50

    # PDHG parameters
    convergence_window::Int = 200
    convergence_check::Int = 50
    max_iter::Int = 0
    min_iter::Int = 40
    divergence_min_update::Int = 50
    max_iter_lp::Int = 10_000_000
    max_iter_conic::Int = 1_000_000
    max_iter_local::Int = 0 #ignores user setting

    advanced_initialization::Bool = true

    # Linesearch parameters
    line_search_flag::Bool = true
    max_linsearch_steps::Int = 5000
    delta::Float64 = .9999
    initial_theta::Float64 = 1.
    linsearch_decay::Float64 = .75

    # Spectral decomposition parameters
    full_eig_decomp::Bool = false
    max_target_rank_krylov_eigs::Int = 16
    min_size_krylov_eigs::Int = 100
    warm_start_eig::Bool = true
    rank_increment::Int = 1 # 0=multiply, 1 = add
    rank_increment_factor::Int = 1 # 0 multiply, 1 = add

    # eigsolver selection
    #=
        1: Arpack [dsaupd] (tipically non-deterministic)
        2: KrylovKit [eigsolve/Lanczos] (DEFAULT)
    =#
    eigsolver::Int = 2
    eigsolver_min_lanczos::Int = 25
    eigsolver_resid_seed::Int = 1234

    # Arpack
    # note that Arpack is Non-deterministic
    # (https://github.com/mariohsouto/ProxSDP.jl/issues/69)
    arpack_tol::Float64 = 1e-10
    #=
        0: arpack random [usually faster - NON-DETERMINISTIC - slightly]
        1: all ones [???]
        2: julia random uniform (eigsolver_resid_seed) [medium for DETERMINISTIC]
        3: julia normalized random normal (eigsolver_resid_seed) [best for DETERMINISTIC]
    =#
    arpack_resid_init::Int = 3
    arpack_reset_resid::Bool = true # true for determinism
    # larger is more stable to converge and more deterministic
    arpack_max_iter::Int = 10_000
    # see remark for of dsaupd

    # KrylovKit
    krylovkit_reset_resid::Bool = false
    krylovkit_resid_init::Int = 3
    krylovkit_tol::Float64 = 1e-12
    krylovkit_max_iter::Int = 100
    krylovkit_eager::Bool = false
    krylovkit_verbose::Int = 0

    # Reduce rank [warning: heuristics]
    reduce_rank::Bool = false
    rank_slack::Int = 3

    full_eig_freq::Int = 10_000_000
    full_eig_len::Int = 0

    # equilibration parameters
    equilibration::Bool = false
    equilibration_iters::Int = 1000
    equilibration_lb::Float64 = -10.0
    equilibration_ub::Float64 = +10.0
    equilibration_limit::Float64 = 0.9
    equilibration_force::Bool = false

    # spectral norm [using exact norm via svds may result in nondeterministic behavior]
    approx_norm::Bool = true
end
