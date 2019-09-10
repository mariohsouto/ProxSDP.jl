function build_simple_lp!(pre_opt::MOIU.CachingOptimizer)
    optim = MOIB.full_bridge_optimizer(pre_opt, Float64)
    MOI.empty!(optim)
    @test MOI.is_empty(optim)

    # add 10 variables - only diagonal is relevant
    X = MOI.add_variables(optim, 2)

    # add sdp constraints - only ensuring positivenesse of the diagonal
    vov = MOI.VectorOfVariables(X)

    c1 = MOI.add_constraint(optim, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(2.0, X[1]),
            MOI.ScalarAffineTerm(1.0, X[2])
        ], 0.0), MOI.EqualTo(4.0))

    c2 = MOI.add_constraint(optim, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1]),
            MOI.ScalarAffineTerm(2.0, X[2])
        ], 0.0), MOI.EqualTo(4.0))

    b1 = MOI.add_constraint(optim, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[1])
        ], 0.0), MOI.GreaterThan(0.0))

    b2 = MOI.add_constraint(optim, 
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, X[2])
        ], 0.0), MOI.GreaterThan(0.0))

    MOI.set(optim, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-4.0, -3.0], [X[1], X[2]]), 0.0)
        )
    MOI.set(optim, MOI.ObjectiveSense(), MOI.MIN_SENSE)
end

@testset "MOI status" begin
    @testset "MOI.OPTIMAL" begin
        build_simple_lp!(optimizer)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        MOI.empty!(optimizer)
    end

    @testset "MOI.ITERATION_LIMIT" begin
        build_simple_lp!(optimizer_maxiter)
        MOI.optimize!(optimizer_maxiter)
        @test MOI.get(optimizer_maxiter, MOI.TerminationStatus()) == MOI.ITERATION_LIMIT
        MOI.empty!(optimizer_maxiter)
    end

    @testset "MOI.ITERATION_LIMIT" begin
        build_simple_lp!(optimizer_timelimit)
        MOI.optimize!(optimizer_timelimit)
        @test MOI.get(optimizer_timelimit, MOI.TerminationStatus()) == MOI.TIME_LIMIT
        MOI.empty!(optimizer_timelimit)
    end
end