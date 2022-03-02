function simple_lp(bridged)

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

function simple_lp_2_1d_sdp(bridged)

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

function lp_in_SDP_equality_form(bridged)

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

function lp_in_SDP_inequality_form(optimizer)

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

end

function sdp_from_moi(optimizer)
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

function double_sdp_from_moi(optimizer)
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

function double_sdp_with_duplicates(optimizer)

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

function sdp_wiki(optimizer)
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


simple_lp(optimizer_bridged)
simple_lp_2_1d_sdp(optimizer_bridged)
lp_in_SDP_equality_form(optimizer_bridged)
lp_in_SDP_inequality_form(optimizer_bridged)
sdp_from_moi(optimizer_bridged)
double_sdp_from_moi(optimizer_bridged)
double_sdp_with_duplicates(optimizer_bridged)
sdp_wiki(optimizer_bridged)

# print test
const optimizer_print = MOI.instantiate(
    ()->ProxSDP.Optimizer(
        log_freq = 10, log_verbose = true, timer_verbose = true, extended_log = true, extended_log2 = true,
        tol_gap = 1e-4, tol_feasibility = 1e-4),
    with_bridge_type = Float64)
sdp_wiki(optimizer_print)

# eig solvers
default_solver = MOI.get(optimizer_bridged, MOI.RawOptimizerAttribute("eigsolver"))
MOI.set(optimizer_bridged, MOI.RawOptimizerAttribute("eigsolver"), 1)
sdp_wiki(optimizer_bridged)
MOI.set(optimizer_bridged, MOI.RawOptimizerAttribute("eigsolver"), 2)
sdp_wiki(optimizer_bridged)
MOI.set(optimizer_bridged, MOI.RawOptimizerAttribute("eigsolver"), default_solver)
MOI.set(optimizer_bridged, MOI.RawOptimizerAttribute("full_eig_decomp"), true)
sdp_wiki(optimizer_bridged)
MOI.set(optimizer_bridged, MOI.RawOptimizerAttribute("full_eig_decomp"), false)
