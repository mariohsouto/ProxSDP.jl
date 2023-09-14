push!(Base.LOAD_PATH, joinpath(dirname(@__FILE__), "..", ".."))

using Test
import ProxSDP
import MathOptInterface
import LinearAlgebra
import Random
import SparseArrays
import DelimitedFiles

const MOI = MathOptInterface
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

const optimizer_bridged = MOI.instantiate(
    ()->ProxSDP.Optimizer(
        tol_gap = 1e-6, tol_feasibility= 1e-6,
        # max_iter = 100_000,
        time_limit = 3., #seconds FAST
        warn_on_limit = true,
        # log_verbose = true, log_freq = 100000
        ),
    with_bridge_type = Float64)

@testset "SolverName" begin
    @test MOI.get(optimizer_bridged, MOI.SolverName()) == "ProxSDP"
end

@testset "SolverVersion" begin
    ver = readlines(joinpath(@__DIR__, "..", "Project.toml"))[4][12:16]
    @test MOI.get(optimizer_bridged, MOI.SolverVersion()) == ver
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
            MOI.ScalarFunctionConstantNotZero, # ignored by UniversalFallback
        ],
    )

    opt = ProxSDP.Optimizer(
        tol_gap = 1e-6, tol_feasibility= 1e-6,
        # max_iter = 100_000,
        time_limit = 1.0, #seconds FAST
        warn_on_limit = true,
        # log_verbose = true, log_freq = 100000
        )
    model = MOI.instantiate(()->opt, with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true)

    MOI.Test.runtests(
        model,
        config,
        exclude = String[
            # Not ProxSDP fail.
            # see: https://github.com/jump-dev/MathOptInterface.jl/issues/1665
            "test_model_UpperBoundAlreadySet",
            "test_model_LowerBoundAlreadySet",
            # poorly scaled problem (solved bellow with higher accuracy)
            "test_linear_add_constraints",
            # time limit hit
            "test_linear_INFEASIBLE",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan_max",
            # TODO(odow): investigate
            "test_HermitianPSDCone_basic",
        ],
    )

    MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), 5.0)
    MOI.empty!(model)
    MOI.Test.test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan_max(
        model,
        config,
    )
    MOI.empty!(model)
    MOI.Test.test_linear_INFEASIBLE(
        model,
        config,
    )

    MOI.set(model, MOI.RawOptimizerAttribute("tol_primal"), 1e-7)
    MOI.set(model, MOI.RawOptimizerAttribute("tol_dual"), 1e-7)
    MOI.set(model, MOI.RawOptimizerAttribute("tol_gap"), 1e-7)
    MOI.set(model, MOI.RawOptimizerAttribute("tol_feasibility"), 1e-7)

    MOI.set(model, MOI.Silent(), true)
    MOI.empty!(model)
    MOI.Test.test_linear_add_constraints(
        model,
        config,
    )

    @test MOI.get(model, ProxSDP.PDHGIterations()) >= 0

    return
end

@testset "MOI Unit" begin
    test_runtests()
end

@testset "ProxSDP MOI Units tests" begin
    include("moi_proxsdp_unit.jl")
end

@testset "MIMO Sizes" begin
    include("base_mimo.jl")
    include("moi_mimo.jl")
    for i in 2:5
        @testset "MIMO n = $(i)" begin
            moi_mimo(optimizer_bridged, 123, i, test = true)
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
opt_low_acc = ProxSDP.Optimizer(
    tol_gap = 1e-3, tol_feasibility = 1e-3,
    # log_verbose = true, log_freq = 1000
    )
cache = MOI.default_cache(opt_low_acc, Float64)
optimizer_low_acc = MOI.Utilities.CachingOptimizer(cache, opt_low_acc)
@testset "SDPLIB Sizes" begin
    datapath = joinpath(dirname(@__FILE__), "data")
    include("base_sdplib.jl")
    include("moi_sdplib.jl")
    @testset "EQPART" begin
        # badly conditioned
        # moi_sdplib(optimizer_low_acc, joinpath(datapath, "gpp124-1.dat-s"), test = true)
        moi_sdplib(optimizer_low_acc, joinpath(datapath, "gpp124-2.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "gpp124-3.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "gpp124-4.dat-s"), test = true)
    end
    @testset "MAX CUT" begin
        moi_sdplib(optimizer_low_acc, joinpath(datapath, "mcp124-1.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "mcp124-2.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "mcp124-3.dat-s"), test = true)
        # moi_sdplib(optimizer, joinpath(datapath, "mcp124-4.dat-s"), test = true)
    end
end

@testset "Sensor Localization" begin
    include("base_sensorloc.jl")
    include("moi_sensorloc.jl")
    for n in 5:5:10
        moi_sensorloc(optimizer_bridged, 0, n, test = true)
    end
end

@testset "Unsupported argument" begin
    @test_throws ErrorException optimizer_unsupportedarg = ProxSDP.Optimizer(unsupportedarg = 10)
    # @test_throws ErrorException MOI.optimize!(optimizer_unsupportedarg)
end

include("test_terminationstatus.jl")

@testset "test_attribute_TimeLimitSec" begin
    model = ProxSDP.Optimizer()
    @test MOI.supports(model, MOI.TimeLimitSec())
    @test MOI.get(model, MOI.TimeLimitSec()) === nothing
    MOI.set(model, MOI.TimeLimitSec(), 0.0)
    @test MOI.get(model, MOI.TimeLimitSec()) == 0.0
    MOI.set(model, MOI.TimeLimitSec(), nothing)
    @test MOI.get(model, MOI.TimeLimitSec()) === nothing
    MOI.set(model, MOI.TimeLimitSec(), 1.0)
    @test MOI.get(model, MOI.TimeLimitSec()) == 1.0
    return
end
