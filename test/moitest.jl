push!(Base.LOAD_PATH, joinpath(dirname(@__FILE__), "..", ".."))

using Test
import ProxSDP
import MathOptInterface
import LinearAlgebra
import Random
import SparseArrays
import DelimitedFiles

const MOI = MathOptInterface
const MOIT = MOI.DeprecatedTest
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

const cache = MOIU.UniversalFallback(MOIU.Model{Float64}())

const optimizer = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(
        tol_gap = 1e-6, tol_feasibility= 1e-6,
        # max_iter = 100_000,
        time_limit = 3., #seconds FAST
        warn_on_limit = true,
        # log_verbose = true, log_freq = 100000
        ))
const optimizer_slow = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(
        tol_gap = 1e-6, tol_feasibility= 1e-6,
        # max_iter = 100_000,
        time_limit = 30., #seconds
        warn_on_limit = true,
        # log_verbose = true, log_freq = 100000
        ))
const optimizer_high_acc = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(
        tol_primal = 1e-7, tol_dual = 1e-7,
        tol_gap = 1e-7, tol_feasibility = 1e-7,
        # log_verbose = true, log_freq = 1000
        ))
const optimizer_low_acc = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(
        tol_gap = 1e-3, tol_feasibility = 1e-3,
        # log_verbose = true, log_freq = 1000
        ))
const optimizer_full = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(full_eig_decomp = true, tol_gap = 1e-4, tol_feasibility = 1e-4))
const optimizer_print = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(log_freq = 10, log_verbose = true, timer_verbose = true, extended_log = true, extended_log2 = true,
    tol_gap = 1e-4, tol_feasibility = 1e-4))
const optimizer_lowacc_arpack = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(eigsolver = 1, tol_gap = 1e-3, tol_feasibility = 1e-3, log_verbose = false))
const optimizer_lowacc_krylovkit = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(eigsolver = 2, tol_gap = 1e-3, tol_feasibility = 1e-3, log_verbose = false))
const config = MOIT.Config{Float64}(atol=1e-3, rtol=1e-3, infeas_certificates = true)
const config_conic = MOIT.Config{Float64}(atol=1e-3, rtol=1e-3, duals = true, infeas_certificates = true)

@testset "SolverName" begin
    @test MOI.get(optimizer, MOI.SolverName()) == "ProxSDP"
end

@testset "SolverVersion" begin
    ver = readlines(joinpath(@__DIR__, "..", "Project.toml"))[4][12:16]
    @test MOI.get(optimizer, MOI.SolverVersion()) == ver
end

function test_runtests()

    config = MOI.Test.Config(
        atol = 1e-4,
        rtol = 1e-3,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
            MOI.ScalarFunctionConstantNotZero, # can be ignored due UniversalFallback
        ],
    )

    opt = ProxSDP.Optimizer(
        tol_gap = 1e-6, tol_feasibility= 1e-6,
        # max_iter = 100_000,
        time_limit = 1.0, #seconds FAST
        warn_on_limit = true,
        # log_verbose = true, log_freq = 100000
        )

    MOI.set(opt, MOI.Silent(), true)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    @testset "fixme" begin
        obj_attr = MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
        @test MOI.supports(model, obj_attr)
        x = MOI.add_variable(model)
        f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(0.0, x)], 0.0)
        MOI.set(model, obj_attr, f)
        MOI.optimize!(model)
        @test_broken MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test_broken MOI.get(model, MOI.ObjectiveValue()) == 0.0
    end
    MOI.Test.runtests(
        model,
        config,
        exclude = String[
            # unexpected failure. But this is probably in the bridge
            # layer, not ProxSDP.
            # see: https://github.com/jump-dev/MathOptInterface.jl/issues/1665
            "test_model_UpperBoundAlreadySet",
            "test_model_LowerBoundAlreadySet",
            # TODO(joaquimg): good catch, but very pathological
            "test_objective_ObjectiveFunction_blank",
            # poorly scaled problem (solved bellow with higher accuracy)
            "test_linear_add_constraints",
        ],
    )

    opt = ProxSDP.Optimizer(
        tol_primal = 1e-7, tol_dual = 1e-7,
        tol_gap = 1e-7, tol_feasibility = 1e-7,
        time_limit = 5.0,
        warn_on_limit = true,
        )
    MOI.set(opt, MOI.Silent(), true)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    MOI.empty!(model)
    MOI.Test.test_linear_add_constraints(
        model,
        config,
    )

    return
end

@testset "MOI Unit" begin
    test_runtests()
end

@testset "Old MOI Unit" begin
    bridged = MOIB.full_bridge_optimizer(optimizer, Float64)
    MOIT.unittest(bridged, config,[
        # not supported attributes
        "number_threads",
        # Quadratic functions are not supported
        "solve_qcp_edge_cases", "solve_qp_edge_cases",
        # Integer and ZeroOne sets are not supported
        "solve_integer_edge_cases", "solve_objbound_edge_cases",
        "solve_zero_one_with_bounds_1",
        "solve_zero_one_with_bounds_2",
        "solve_zero_one_with_bounds_3",
        # farkas proof
        "solve_farkas_interval_upper",
        "solve_farkas_interval_lower",
        "solve_farkas_equalto_upper",
        "solve_farkas_equalto_lower",
        "solve_farkas_variable_lessthan_max",
        "solve_farkas_variable_lessthan",
        "solve_farkas_lessthan",
        "solve_farkas_greaterthan",
        ]
    )
    # TODO:
    # bridged_slow = MOIB.full_bridge_optimizer(optimizer_slow, Float64)
    # MOIT.solve_farkas_interval_upper(bridged_slow, config)
    # MOIT.solve_farkas_interval_lower(bridged, config)
    # MOIT.solve_farkas_equalto_upper(bridged_slow, config)
    # MOIT.solve_farkas_equalto_lower(bridged, config)
    # MOIT.solve_farkas_variable_lessthan_max(bridged_slow, config)
    # MOIT.solve_farkas_variable_lessthan(bridged_slow, config)
    # MOIT.solve_farkas_lessthan(bridged_slow, config)
    # MOIT.solve_farkas_greaterthan(bridged, config)
end

@testset "MOI Continuous Linear" begin
    bridged = MOIB.full_bridge_optimizer(optimizer, Float64)
    # MOIT.linear8atest(MOIB.full_bridge_optimizer(optimizer_high_acc, Float64), config)
    # MOIT.linear8btest(MOIB.full_bridge_optimizer(optimizer_high_acc, Float64), config)
    # MOIT.linear8ctest(MOIB.full_bridge_optimizer(optimizer_high_acc, Float64), config)
    # MOIT.linear12test(MOIB.full_bridge_optimizer(optimizer_high_acc, Float64), config)
    MOIT.contlineartest(bridged, config, [
        # infeasible/unbounded
        "linear8a",
        #"linear8b", "linear8c", "linear12",
        # poorly conditioned
        "linear10",
        "linear5",
        "linear9",
        # primalstart not accepted
        "partial_start",
        ]
    )
    MOIT.linear8atest(MOIB.full_bridge_optimizer(optimizer_high_acc, Float64), config)
    MOIT.linear9test(MOIB.full_bridge_optimizer(optimizer_high_acc, Float64), config)
    MOIT.linear5test(MOIB.full_bridge_optimizer(optimizer_high_acc, Float64), config)
    MOIT.linear10test(MOIB.full_bridge_optimizer(optimizer_high_acc, Float64), config)
end

@testset "MOI Continuous Conic" begin
    MOIT.contconictest(MOIB.full_bridge_optimizer(optimizer, Float64), config_conic, [
        # bridge: some problem with square psd
        "rootdets",
        # exp cone
        "logdet", "exp", "dualexp",
        # pow cone
        "pow","dualpow",
        # other cones
        "relentr",
        # infeasible/unbounded
        # "lin3", "lin4",
        # See https://travis-ci.com/blegat/SolverTests/jobs/268551133
        # geomean2v: Test Failed at /home/travis/.julia/dev/MathOptInterface/src/Test/contconic.jl:1328
        # Expression: MOI.get(model, MOI.TerminationStatus()) == config.optimal_status
        #  Evaluated: INFEASIBLE_OR_UNBOUNDED::TerminationStatusCode = 6 == OPTIMAL::TerminationStatusCode = 1
        # "geomean2v", "geomean2f", , "rotatedsoc2", "psdt2", 
        # "normone2", "norminf2", "rotatedsoc2"#
        # slow to find certificate
        "normone2",
        "norminf2", # new
        ]
    )
    # # these fail due to infeasibility certificate not being disabled
    # MOIT.norminf2test(MOIB.full_bridge_optimizer(optimizer, Float64), config_conic_nodual)
    # MOIT.normone2test(MOIB.full_bridge_optimizer(optimizer_slow, Float64), config_conic)
    # # requires certificates always
    # MOIT.rotatedsoc2test(MOIB.full_bridge_optimizer(optimizer, Float64), config_conic_nodual)
end

@testset "ProxSDP MOI Units tests" begin
    include("moi_proxsdp_unit.jl")
end

@testset "MIMO Sizes" begin
    include("base_mimo.jl")
    include("moi_mimo.jl")
    for i in 2:5
        @testset "MIMO n = $(i)" begin 
            moi_mimo(optimizer, 123, i, test = true)
        end
    end
end

# hitting time limit
# probably infeasible/unbounded
# @testset "RANDSDP Sizes" begin
#     include("base_randsdp.jl")
#     include("moi_randsdp.jl")
#     for n in 10:11, m in 10:11
#         @testset "RANDSDP n=$n, m=$m" begin 
#             moi_randsdp(optimizer, 123, n, m, test = true, atol = 1e-1)
#         end
#     end
# end

# This problems are too large for Travis
@testset "SDPLIB Sizes" begin
    datapath = joinpath(dirname(@__FILE__), "data")
    include("base_sdplib.jl")
    include("moi_sdplib.jl")
    @testset "EQPART" begin
        # badly conditioned
        # moi_sdplib(optimizer_low_acc, joinpath(datapath, "gpp124-1.dat-s"), test = true)
        moi_sdplib(optimizer_low_acc, joinpath(datapath, "gpp124-2.dat-s"), test = true)
        moi_sdplib(optimizer_lowacc_arpack, joinpath(datapath, "gpp124-2.dat-s"), test = true)
        moi_sdplib(optimizer_lowacc_krylovkit, joinpath(datapath, "gpp124-2.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "gpp124-3.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "gpp124-4.dat-s"), test = true)
    end
    @testset "MAX CUT" begin
        moi_sdplib(optimizer_low_acc, joinpath(datapath, "mcp124-1.dat-s"), test = true)
        moi_sdplib(optimizer_lowacc_arpack, joinpath(datapath, "mcp124-1.dat-s"), test = true)
        moi_sdplib(optimizer_lowacc_krylovkit, joinpath(datapath, "mcp124-1.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "mcp124-2.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "mcp124-3.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "mcp124-4.dat-s"), test = true)
    end
end

@testset "Sensor Localization" begin
    include("base_sensorloc.jl")
    include("moi_sensorloc.jl")
    for n in 5:5:10
        moi_sensorloc(optimizer, 0, n, test = true)
    end
end

@testset "Full eig" begin
    MOIT.psdt0vtest(
        MOIB.full_bridge_optimizer(optimizer_full, Float64),
        MOIT.Config{Float64}(atol=1e-3, rtol=1e-3, duals = false)
        )
end

@testset "Print" begin
    MOIT.linear15test(optimizer_print, MOIT.Config{Float64}(atol=1e-3, rtol=1e-3))
end

@testset "Unsupported argument" begin
    MOI.empty!(cache)
    @test_throws ErrorException optimizer_unsupportedarg = MOIU.CachingOptimizer(cache, ProxSDP.Optimizer(unsupportedarg = 10))
    # @test_throws ErrorException MOI.optimize!(optimizer_unsupportedarg)
end

include("test_terminationstatus.jl")
