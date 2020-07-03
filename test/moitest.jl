push!(Base.LOAD_PATH,joinpath(dirname(@__FILE__),"..",".."))

using ProxSDP, MathOptInterface, Test, LinearAlgebra, Random, SparseArrays, DelimitedFiles

const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

const cache = MOIU.UniversalFallback(MOIU.Model{Float64}())

const optimizer = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(tol_primal = 1e-6, tol_dual = 1e-6, log_verbose = false))
const optimizer_low_acc = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(tol_primal = 1e-3, tol_dual = 1e-3, log_verbose = true, timer_verbose = true))
const optimizer_full = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(full_eig_decomp = true, tol_primal = 1e-4, tol_dual = 1e-4))
const optimizer_print = MOIU.CachingOptimizer(cache,
    ProxSDP.Optimizer(log_freq = 10, log_verbose = true, timer_verbose = true, tol_primal = 1e-4, tol_dual = 1e-4))
const config = MOIT.TestConfig(atol=1e-3, rtol=1e-3, infeas_certificates = false)
const config_conic = MOIT.TestConfig(atol=1e-3, rtol=1e-3, duals = false, infeas_certificates = false)

@testset "SolverName" begin
    @test MOI.get(optimizer, MOI.SolverName()) == "ProxSDP"
end

@testset "supports_default_copy_to" begin
    @test MOIU.supports_allocate_load(ProxSDP.Optimizer(), false)
    @test !MOIU.supports_allocate_load(ProxSDP.Optimizer(), true)
end

@testset "Unit" begin
    bridged = MOIB.full_bridge_optimizer(optimizer, Float64)
    MOIT.unittest(bridged, config,[
        # Quadratic functions are not supported
        "solve_qcp_edge_cases", "solve_qp_edge_cases",
        # Integer and ZeroOne sets are not supported
        "solve_integer_edge_cases", "solve_objbound_edge_cases",
        "solve_zero_one_with_bounds_1",
        "solve_zero_one_with_bounds_2",
        "solve_zero_one_with_bounds_3",
                    # `TimeLimitSec` not supported.
                    # "time_limit_sec",
                    "number_threads",
                    # ArgumentError: The number of constraints in SCSModel must be greater than 0
                    "solve_unbounded_model",
        ]
    )
end

@testset "MOI Continuous Linear" begin
    bridged = MOIB.full_bridge_optimizer(optimizer, Float64)
    MOIT.contlineartest(bridged, config, [
        # infeasible/unbounded
        # "linear8a", "linear8b", "linear8c", "linear12",
        # linear10 is poorly conditioned
        "linear10",
        # linear9 is requires precision
        "linear9",
        # primalstart not accepted
        "partial_start",
        ]
    )
    # MOIT.linear9test(MOIB.SplitInterval{Float64}(optimizer), config)
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
        "geomean2v", "geomean2f",
        "soc3", "psdt2", "normone2", "norminf2",#, "rotatedsoc2"
        ]
    )
end

@testset "Simple LP" begin

    bridged = MOIB.full_bridge_optimizer(optimizer, Float64)
    MOI.empty!(bridged)
    @test MOI.is_empty(bridged)

    # add 10 variables - only diagonal is relevant
    X = MOI.add_variables(bridged, 2)

    # add sdp constraints - only ensuring positivenesse of the diagonal
    vov = MOI.VectorOfVariables(X)

    c1 = MOI.add_constraint(bridged, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(2.0, X[1]),
            MOI.ScalarAffineTerm(1.0, X[2])
        ], 0.0), MOI.EqualTo(4.0))

    c2 = MOI.add_constraint(bridged, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1]),
            MOI.ScalarAffineTerm(2.0, X[2])
        ], 0.0), MOI.EqualTo(4.0))

    b1 = MOI.add_constraint(bridged, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1])
        ], 0.0), MOI.GreaterThan(0.0))

    b2 = MOI.add_constraint(bridged, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[2])
        ], 0.0), MOI.GreaterThan(0.0))

    MOI.set(bridged, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-4.0, -3.0], [X[1], X[2]]), 0.0)
        )
    MOI.set(bridged, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(bridged)

    obj = MOI.get(bridged, MOI.ObjectiveValue())

    @test obj ≈ -9.33333 atol = 1e-2

    Xr = MOI.get(bridged, MOI.VariablePrimal(), X)

    @test Xr ≈ [1.3333, 1.3333] atol = 1e-2

end

@testset "Simple LP with 2 1D SDP" begin

    bridged = MOIB.full_bridge_optimizer(optimizer, Float64)
    MOI.empty!(bridged)
    @test MOI.is_empty(bridged)

    # add 10 variables - only diagonal is relevant
    X = MOI.add_variables(bridged, 2)

    # add sdp constraints - only ensuring positivenesse of the diagonal
    vov = MOI.VectorOfVariables(X)

    c1 = MOI.add_constraint(bridged, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(2.0, X[1]),
            MOI.ScalarAffineTerm(1.0, X[2])
        ], 0.0), MOI.EqualTo(4.0))

    c2 = MOI.add_constraint(bridged, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1]),
            MOI.ScalarAffineTerm(2.0, X[2])
        ], 0.0), MOI.EqualTo(4.0))

    b1 = MOI.add_constraint(bridged, 
        MOI.VectorOfVariables([X[1]]), MOI.PositiveSemidefiniteConeTriangle(1))

    b2 = MOI.add_constraint(bridged, 
        MOI.VectorOfVariables([X[2]]), MOI.PositiveSemidefiniteConeTriangle(1))

    MOI.set(bridged, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-4.0, -3.0], [X[1], X[2]]), 0.0)
        )
    MOI.set(bridged, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(bridged)

    obj = MOI.get(bridged, MOI.ObjectiveValue())

    @test obj ≈ -9.33333 atol = 1e-2

    Xr = MOI.get(bridged, MOI.VariablePrimal(), X)

    @test Xr ≈ [1.3333, 1.3333] atol = 1e-2

end

@testset "LP in SDP EQ form" begin

    bridged = MOIB.full_bridge_optimizer(optimizer, Float64)
    MOI.empty!(bridged)
    @test MOI.is_empty(bridged)

    # add 10 variables - only diagonal is relevant
    X = MOI.add_variables(bridged, 10)

    # add sdp constraints - only ensuring positivenesse of the diagonal
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(bridged, vov, MOI.PositiveSemidefiniteConeTriangle(4))

    c1 = MOI.add_constraint(bridged, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(2.0, X[1]),
            MOI.ScalarAffineTerm(1.0, X[3]),
            MOI.ScalarAffineTerm(1.0, X[6])
        ], 0.0), MOI.EqualTo(4.0))

    c2 = MOI.add_constraint(bridged, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1]),
            MOI.ScalarAffineTerm(2.0, X[3]),
            MOI.ScalarAffineTerm(1.0, X[10])
        ], 0.0), MOI.EqualTo(4.0))

    MOI.set(bridged, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-4.0, -3.0], [X[1], X[3]]), 0.0)
        )
    MOI.set(bridged, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(bridged)

    obj = MOI.get(bridged, MOI.ObjectiveValue())

    @test obj ≈ -9.33333 atol = 1e-2

    Xr = MOI.get(bridged, MOI.VariablePrimal(), X)

    @test Xr ≈ [1.3333, .0, 1.3333, .0, .0, .0, .0, .0, .0, .0] atol = 1e-2

end

@testset "LP in SDP INEQ form" begin

    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    # add 10 variables - only diagonal is relevant
    X = MOI.add_variables(optimizer, 3)

    # add sdp constraints - only ensuring positivenesse of the diagonal
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(2))

    c1 = MOI.add_constraint(optimizer, 
        MOI.VectorAffineFunction(MOI.VectorAffineTerm.([1,1],[
            MOI.ScalarAffineTerm(2.0, X[1]),
            MOI.ScalarAffineTerm(1.0, X[3]),
        ]), [-4.0]), MOI.Nonpositives(1))

    c2 = MOI.add_constraint(optimizer, 
        MOI.VectorAffineFunction(MOI.VectorAffineTerm.([1,1],[
            MOI.ScalarAffineTerm(1.0, X[1]),
            MOI.ScalarAffineTerm(2.0, X[3]),
        ]), [-4.0]), MOI.Nonpositives(1))

    MOI.set(optimizer, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([4.0, 3.0], [X[1], X[3]]), 0.0)
        )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())

    @test obj ≈ 9.33333 atol = 1e-2

    Xr = MOI.get(optimizer, MOI.VariablePrimal(), X)

    @test Xr ≈ [1.3333, .0, 1.3333] atol = 1e-2

    c1_d = MOI.get(optimizer, MOI.ConstraintDual(), c1)
    c2_d = MOI.get(optimizer, MOI.ConstraintDual(), c2)

    # SDP duasl error
    @test_throws ErrorException c2_d = MOI.get(optimizer, MOI.ConstraintDual(), cX)

end

@testset "SDP from MOI" begin
    # min X[1,1] + X[2,2]    max y
    #     X[2,1] = 1         [0   y/2     [ 1  0
    #                         y/2 0    <=   0  1]
    #     X >= 0              y free
    # Optimal solution:
    #
    #     ⎛ 1   1 ⎞
    # X = ⎜       ⎟           y = 2
    #     ⎝ 1   1 ⎠
    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    X = MOI.add_variables(optimizer, 3)

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(2))

    c = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0, X[2]))], [-1.0]), MOI.Zeros(1))

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [X[1], X[3]]), 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL

    @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(optimizer, MOI.DualStatus()) == MOI.FEASIBLE_POINT

    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 2 atol=1e-2

    Xv = ones(3)
    @test MOI.get(optimizer, MOI.VariablePrimal(), X) ≈ Xv atol=1e-2
    # @test MOI.get(optimizer, MOI.ConstraintPrimal(), cX) ≈ Xv atol=1e-2

    # @test MOI.get(optimizer, MOI.ConstraintDual(), c) ≈ 2 atol=1e-2
    # @show MOI.get(optimizer, MOI.ConstraintDual(), c)

end

@testset "Double SDP from MOI" begin
    # solve simultaneously two of these:
    # min X[1,1] + X[2,2]    max y
    #     X[2,1] = 1         [0   y/2     [ 1  0
    #                         y/2 0    <=   0  1]
    #     X >= 0              y free
    # Optimal solution:
    #
    #     ⎛ 1   1 ⎞
    # X = ⎜       ⎟           y = 2
    #     ⎝ 1   1 ⎠
    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    X = MOI.add_variables(optimizer, 3)
    Y = MOI.add_variables(optimizer, 3)

    vov = MOI.VectorOfVariables(X)
    vov2 = MOI.VectorOfVariables(Y)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(2))
    cY = MOI.add_constraint(optimizer, vov2, MOI.PositiveSemidefiniteConeTriangle(2))

    c = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0, X[2]))], [-1.0]), MOI.Zeros(1))
    c2 = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0, Y[2]))], [-1.0]), MOI.Zeros(1))

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [X[1], X[end], Y[1], Y[end]]), 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL

    @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(optimizer, MOI.DualStatus()) == MOI.FEASIBLE_POINT

    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 2*2 atol=1e-2

    Xv = ones(3)
    @test MOI.get(optimizer, MOI.VariablePrimal(), X) ≈ Xv atol=1e-2
    Yv = ones(3)
    @test MOI.get(optimizer, MOI.VariablePrimal(), Y) ≈ Yv atol=1e-2
    # @test MOI.get(optimizer, MOI.ConstraintPrimal(), cX) ≈ Xv atol=1e-2

    # @test MOI.get(optimizer, MOI.ConstraintDual(), c) ≈ 2 atol=1e-2
    # @show MOI.get(optimizer, MOI.ConstraintDual(), c)

end

@testset "SDP with duplicates from MOI" begin

    using MathOptInterface
    # using SCS,ProxSDP
    MOI = MathOptInterface
    MOIU = MathOptInterface.Utilities
    MOIB = MathOptInterface.Bridges

    cache = MOIU.UniversalFallback(MOIU.Model{Float64}());
    #optimizer0 = SCS.Optimizer(linear_solver=SCS.Direct, eps=1e-8);
    optimizer0 = ProxSDP.Optimizer()#linear_solver=SCS.Direct, eps=1e-8);
    MOI.empty!(cache);
    optimizer1 = MOIU.CachingOptimizer(cache, optimizer0);
    optimizer = MOIB.full_bridge_optimizer(optimizer1, Float64);

    MOI.empty!(optimizer)

    x = MOI.add_variable(optimizer)
    X = [x, x, x]

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(2))

    c = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0, X[2]))], [-1.0]), MOI.Zeros(1))

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [X[1], X[3]]), 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL

    @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(optimizer, MOI.DualStatus()) == MOI.FEASIBLE_POINT

    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 2 atol=1e-2

    Xv = ones(3)
    @test MOI.get(optimizer, MOI.VariablePrimal(), X) ≈ Xv atol=1e-2

end

@testset "SDP from Wikipedia" begin
    # https://en.wikipedia.org/wiki/Semidefinite_programming
    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    X = MOI.add_variables(optimizer, 6)

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(3))

    cd1 = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0, X[1]))], [-1.0]), MOI.Zeros(1))
    cd1 = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0, X[3]))], [-1.0]), MOI.Zeros(1))
    cd1 = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0, X[6]))], [-1.0]), MOI.Zeros(1))

    c12_ub = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0, X[2]))], [0.1]), MOI.Nonpositives(1))   # x <= -0.1 -> x + 0.1 <= 0
    c12_lb = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(-1.0, X[2]))], [-0.2]), MOI.Nonpositives(1)) # x >= -0.2 -> -x + -0.2 <= 0

    c23_ub = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0, X[5]))], [-0.5]), MOI.Nonpositives(1)) # x <= 0.5 ->  x - 0.5 <= 0
    c23_lb = MOI.add_constraint(optimizer, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(-1.0, X[5]))], [0.4]), MOI.Nonpositives(1)) # x >= 0.4 -> -x + 0.4 <= 0

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [X[4]]), 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())
    @test obj ≈ -0.978 atol=1e-2

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())
    @test obj ≈ 0.872 atol=1e-2

end

using LinearAlgebra
LinearAlgebra.symmetric_type(::Type{MathOptInterface.VariableIndex}) = MathOptInterface.VariableIndex
LinearAlgebra.symmetric(v::MathOptInterface.VariableIndex, ::Symbol) = v
LinearAlgebra.transpose(v::MathOptInterface.VariableIndex) = v

@testset "MIMO Sizes" begin
    include("base_mimo.jl")
    include("moi_mimo.jl")
    for i in 2:5
        @testset "MIMO n = $(i)" begin 
            moi_mimo(optimizer, 123, i, test = true)
        end
    end
end

@testset "RANDSDP Sizes" begin
    include("base_randsdp.jl")
    include("moi_randsdp.jl")
    for n in 10:11, m in 10:11
        @testset "RANDSDP n=$n, m=$m" begin 
            moi_randsdp(optimizer, 123, n, m, test = true, atol = 1e-1)
        end
    end
end

# This problems are too large for Travis
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
        moi_sensorloc(optimizer, 0, n, test = true)
    end
end

@testset "Full eig" begin
    MOIT.psdt0vtest(
        MOIB.full_bridge_optimizer(optimizer_full, Float64),
        MOIT.TestConfig(atol=1e-3, rtol=1e-3, duals = false)
        )
end

@testset "Print" begin
    MOIT.linear15test(optimizer_print, MOIT.TestConfig(atol=1e-3, rtol=1e-3))
end

@testset "Unsupported argument" begin
    MOI.empty!(cache)
    @test_throws ErrorException  optimizer_unsupportedarg = MOIU.CachingOptimizer(cache, ProxSDP.Optimizer(unsupportedarg = 10))
    # @test_throws ErrorException MOI.optimize!(optimizer_unsupportedarg)
end

include("test_terminationstatus.jl")
