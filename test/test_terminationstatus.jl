function build_simple_lp!(optimizer::MOIU.CachingOptimizer)
    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    # add 10 variables - only diagonal is relevant
    X = MOI.add_variables(optimizer, 2)

    # add sdp constraints - only ensuring positivenesse of the diagonal
    vov = MOI.VectorOfVariables(X)

    c1 = MOI.add_constraint(optimizer, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(2.0, X[1]),
            MOI.ScalarAffineTerm(1.0, X[2])
        ], 0.0), MOI.EqualTo(4.0))

    c2 = MOI.add_constraint(optimizer, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1]),
            MOI.ScalarAffineTerm(2.0, X[2])
        ], 0.0), MOI.EqualTo(4.0))

    b1 = MOI.add_constraint(optimizer, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1])
        ], 0.0), MOI.GreaterThan(0.0))

    b2 = MOI.add_constraint(optimizer, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[2])
        ], 0.0), MOI.GreaterThan(0.0))

    MOI.set(optimizer, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-4.0, -3.0], [X[1], X[2]]), 0.0)
        )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
end

@testset "MOI status" begin
    @testset "MOI.OPTIMAL" begin
        build_simple_lp!(optimizer)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    end

    @testset "MOI.ITERATION_LIMIT" begin
        build_simple_lp!(optimizer_maxiter)
        MOI.optimize!(optimizer_maxiter)
        @test MOI.get(optimizer_maxiter, MOI.TerminationStatus()) == MOI.ITERATION_LIMIT
    end

    @testset "MOI.ITERATION_LIMIT" begin
        build_simple_lp!(optimizer_timelimit)
        MOI.optimize!(optimizer_timelimit)
        @test MOI.get(optimizer_timelimit, MOI.TerminationStatus()) == MOI.TIME_LIMIT
    end
end