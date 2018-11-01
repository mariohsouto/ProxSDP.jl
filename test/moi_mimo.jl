function moi_mimo(optimizer, seed, n; verbose = false, test = false)

    MOI.empty!(optimizer)
    if test
        @test MOI.is_empty(optimizer)
    end

    m = 10n
    s, H, y, L = mimo_data(seed, m, n)

    nvars = ProxSDP.sympackedlen(n + 1)

    X = MOI.add_variables(optimizer, nvars)

    for i in 1:nvars
        MOI.add_constraint(optimizer, MOI.SingleVariable(X[i]), MOI.LessThan(1.0))
        MOI.add_constraint(optimizer, MOI.SingleVariable(X[i]), MOI.GreaterThan(-1.0))
    end

    Xsq = Matrix{MOI.VariableIndex}(n+1,n+1)
    ProxSDP.ivech!(Xsq, X)
    Xsq = full(Symmetric(Xsq,:U))

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

    if test
        for i in 1:n+1, j in 1:n+1
            @test 1.01> abs(Xsq_s[i,j]) > 0.99
        end
    end

    verbose && mimo_eval(s, H, y, L, Xsq_s)

    return ProxSDP.get_solution(optimizer)
end