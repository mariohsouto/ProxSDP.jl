push!(Base.LOAD_PATH,joinpath(dirname(@__FILE__),"..",".."))

using ProxSDP, MathOptInterface, Base.Test

const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

MOIU.@model ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)

const optimizer = MOIU.CachingOptimizer(ProxSDPModelData{Float64}(), ProxSDP.Optimizer())

# linear9test needs 1e-3 with SCS < 2.0 and 5e-1 with SCS 2.0
# linear2test needs 1e-4
# const config = MOIT.TestConfig(atol=1e-4, rtol=1e-4)

# MOIT._lin1test(optimizer, config, true)

# @testset "Continuous linear problems" begin
#     # AlmostSuccess for linear9 with SCS 2
#     MOIT.contlineartest(MOIB.SplitInterval{Float64}(optimizer), config, ["linear9"])
# end

# @testset "Continuous conic problems" begin
#     MOIT.contconictest(MOIB.RootDet{Float64}(MOIB.LogDet{Float64}(optimizer)), config, ["rsoc", "geomean", "psds", "rootdet", "logdets"])
# end

@testset "LP in SDP EQ form" begin

    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    # add 10 variables - only diagonal is relevant
    X = MOI.add_variables(optimizer, 10)

    # add sdp constraints - only ensuring positivenesse of the diagonal
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, MOI.VectorAffineFunction{Float64}(vov), MOI.PositiveSemidefiniteConeTriangle(4))

    c1 = MOI.add_constraint(optimizer, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(2.0, X[1]),
            MOI.ScalarAffineTerm(1.0, X[3]),
            MOI.ScalarAffineTerm(1.0, X[6])
        ], 0.0), MOI.EqualTo(4.0))

    c2 = MOI.add_constraint(optimizer, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1]),
            MOI.ScalarAffineTerm(2.0, X[3]),
            MOI.ScalarAffineTerm(1.0, X[10])
        ], 0.0), MOI.EqualTo(4.0))

    MOI.set(optimizer, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-4.0, -3.0], [X[1], X[3]]), 0.0)
        )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MinSense)
    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())

    @test obj ≈ -9.33333 atol = 1e-2

    Xr = MOI.get(optimizer, MOI.VariablePrimal(), X)

    @test Xr ≈ [1.3333, .0, 1.3333, .0, .0, .0, .0, .0, .0, .0] atol = 1e-2

end

@testset "LP in SDP INEQ form" begin

    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    # add 10 variables - only diagonal is relevant
    X = MOI.add_variables(optimizer, 3)

    # add sdp constraints - only ensuring positivenesse of the diagonal
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, MOI.VectorAffineFunction{Float64}(vov), MOI.PositiveSemidefiniteConeTriangle(2))

    c1 = MOI.add_constraint(optimizer, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(2.0, X[1]),
            MOI.ScalarAffineTerm(1.0, X[3]),
        ], 0.0), MOI.LessThan(4.0))

    c2 = MOI.add_constraint(optimizer, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1]),
            MOI.ScalarAffineTerm(2.0, X[3]),
        ], 0.0), MOI.LessThan(4.0))

    MOI.set(optimizer, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([4.0, 3.0], [X[1], X[3]]), 0.0)
        )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MaxSense)
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

    c = MOI.add_constraint(optimizer, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, X[2])], 0.0), MOI.EqualTo(1.0))

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [X[1], X[end]]), 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MinSense)
    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.Success

    @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.FeasiblePoint
    @test MOI.get(optimizer, MOI.DualStatus()) == MOI.FeasiblePoint

    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 2 atol=1e-2

    Xv = ones(3)
    @test MOI.get(optimizer, MOI.VariablePrimal(), X) ≈ Xv atol=1e-2
    # @test MOI.get(optimizer, MOI.ConstraintPrimal(), cX) ≈ Xv atol=1e-2

    # @test MOI.get(optimizer, MOI.ConstraintDual(), c) ≈ 2 atol=1e-2
    # @show MOI.get(optimizer, MOI.ConstraintDual(), c)

end

@testset "SDP from Wikipedia" begin
    # https://en.wikipedia.org/wiki/Semidefinite_programming
    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    X = MOI.add_variables(optimizer, 6)

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(3))

    cd1 = MOI.add_constraint(optimizer, MOI.SingleVariable(X[1]), MOI.EqualTo(1.0))
    cd2 = MOI.add_constraint(optimizer, MOI.SingleVariable(X[3]), MOI.EqualTo(1.0))
    cd3 = MOI.add_constraint(optimizer, MOI.SingleVariable(X[6]), MOI.EqualTo(1.0))

    c12_ub = MOI.add_constraint(optimizer, MOI.SingleVariable(X[2]), MOI.LessThan(-0.1))
    c12_lb = MOI.add_constraint(optimizer, MOI.SingleVariable(X[2]), MOI.GreaterThan(-0.2))

    c23_ub = MOI.add_constraint(optimizer, MOI.SingleVariable(X[5]), MOI.LessThan(0.5))
    c23_lb = MOI.add_constraint(optimizer, MOI.SingleVariable(X[5]), MOI.GreaterThan(0.4))

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [X[4]]), 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MinSense)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())
    @test obj ≈ -0.978 atol=1e-2

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MaxSense)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())
    @test obj ≈ 0.872 atol=1e-2

end

@testset "MIMO" begin

    srand(23)

    # Instance size
    n = 3
    m = 10 * n
    # Channel
    H = randn((m, n))
    # Gaussian noise
    v = randn((m, 1))
    # True signal
    s = rand([-1, 1], n)
    # Received signal
    sigma = 10.0
    y = H * s + sigma * v
    L = [hcat(H' * H, -H' * y); hcat(-y' * H, y' * y)]

    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    nvars = ProxSDP.sympackedlen(n+1)

    X = MOI.add_variables(optimizer, nvars)

    @show typeof(X)

    for i in 1:nvars
        MOI.add_constraint(optimizer, MOI.SingleVariable(X[i]), MOI.LessThan(1.0))
        MOI.add_constraint(optimizer, MOI.SingleVariable(X[i]), MOI.GreaterThan(-1.0))
    end

    Xsq = Matrix{MOI.VariableIndex}(n+1,n+1)
    ProxSDP.ivech!(Xsq, X)
    @show Xsq = full(Symmetric(Xsq,:U))

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n+1))


    for i in 1:n+1
        MOI.add_constraint(optimizer, MOI.SingleVariable(Xsq[i,i]), MOI.EqualTo(1.0))
    end

    objf_t = vec([MOI.ScalarAffineTerm(L[i,j], Xsq[i,j]) for i in 1:n+1, j in 1:n+1])
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MinSense)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)

    # for i in 1:n+1, j in 1:n+1
    #     @test 1.01> abs(Xsq_s[i,j]) > 0.99
    # end

end

@testset "MIMO Sizes" begin
    include("base_mimo.jl")
    include("moi_mimo.jl")
    for i in 2:5
        @testset "MIMO n = $(i)" begin 
            moi_mimo(optimizer, 123, i)
        end
    end
end

@testset "RANDSDP Sizes" begin
    include("base_randsdp.jl")
    include("moi_randsdp.jl")
    for n in 10:12, m in 10:12
        @testset "RANDSDP n=$n, m=$m" begin 
            moi_randsdp(optimizer, 123, n, m, atol = 1e-1)
        end
    end
end

@testset "SDPLIB Sizes" begin
    datapath = joinpath(dirname(@__FILE__), "data")
    include("base_sdplib.jl")
    include("moi_sdplib.jl")
    @testset "EQPART" begin
        moi_sdplib(optimizer, joinpath(datapath, "gpp124-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp124-2.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp124-3.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "gpp124-4.dat-s"))
    end
    @testset "MAX CUT" begin
        moi_sdplib(optimizer, joinpath(datapath, "mcp124-1.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp124-2.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp124-3.dat-s"))
        moi_sdplib(optimizer, joinpath(datapath, "mcp124-4.dat-s"))
    end
end