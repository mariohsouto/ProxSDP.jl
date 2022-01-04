function moi_mimo(optimizer, seed, n; verbose = false, test = false, scalar = false)

    MOI.empty!(optimizer)
    if test
        @test MOI.is_empty(optimizer)
    end

    m = 10n
    s, H, y, L = mimo_data(seed, m, n)

    nvars = ProxSDP.sympackedlen(n + 1)

    X = MOI.add_variables(optimizer, nvars)
    if scalar
        for i in 1:nvars
            MOI.add_constraint(optimizer, MOI.SingleVariable(X[i]), MOI.LessThan(1.0))
            MOI.add_constraint(optimizer, MOI.SingleVariable(X[i]), MOI.GreaterThan(-1.0))
        end
    else
        MOI.add_constraint(optimizer,
                            MOI.VectorAffineFunction(
                                MOI.VectorAffineTerm.(
                                    collect(1:nvars), MOI.ScalarAffineTerm.(1.0, X)),
                                -ones(nvars)),
                            MOI.Nonpositives(nvars))
        MOI.add_constraint(optimizer,
                            MOI.VectorAffineFunction(
                                MOI.VectorAffineTerm.(
                                    collect(1:nvars), MOI.ScalarAffineTerm.(-1.0, X)),
                                -ones(nvars)),
                            MOI.Nonpositives(nvars))
    end

    Xsq = Matrix{MOI.VariableIndex}(undef, n+1,n+1)
    ProxSDP.ivech!(Xsq, X)
    Xsq = Matrix(LinearAlgebra.Symmetric(Xsq,:U))

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n+1))
    if scalar
        for i in 1:n+1
            MOI.add_constraint(optimizer, MOI.SingleVariable(Xsq[i,i]), MOI.EqualTo(1.0))
        end
    else
        MOI.add_constraint(optimizer,
                            MOI.VectorAffineFunction(
                                MOI.VectorAffineTerm.(
                                    collect(1:n+1), MOI.ScalarAffineTerm.(1.0, [Xsq[i,i] for i in 1:n+1])),
                                -ones(n+1)),
                            MOI.Zeros(n+1))
    end

    objf_t = vec([MOI.ScalarAffineTerm(L[i,j], Xsq[i,j]) for i in 1:n+1, j in 1:n+1])
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    objval = MOI.get(optimizer, MOI.ObjectiveValue())

    stime = -1.0
    try
        stime = MOI.get(optimizer, MOI.SolveTimeSec())
    catch
        println("could not query time")
    end

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)

    if test
        for i in 1:n+1, j in 1:n+1
            @test 1.01 > abs(Xsq_s[i,j]) > 0.99
        end
    end

    verbose && mimo_eval(s, H, y, L, Xsq_s)

    rank = -1
    status = 0
    if MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        status = 1
    end
    return (objval, stime, rank, status)
end