function moi_mimo(optimizer, seed, n)

    MOI.empty!(optimizer)
    @test MOI.isempty(optimizer)

    # n = 3
    m = 10n
    s, H, y, L = mimo_data(seed, m, n)

    nvars = ProxSDP.sympackedlen(n+1)

    X = MOI.addvariables!(optimizer, nvars)

    for i in 1:nvars
        MOI.addconstraint!(optimizer, MOI.SingleVariable(X[i]), MOI.LessThan(1.0))
        MOI.addconstraint!(optimizer, MOI.SingleVariable(X[i]), MOI.GreaterThan(-1.0))
    end

    Xsq = Matrix{MOI.VariableIndex}(n+1,n+1)
    ProxSDP.ivech!(Xsq, X)
    Xsq = full(Symmetric(Xsq,:U))

    vov = MOI.VectorOfVariables(X)
    cX = MOI.addconstraint!(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n+1))

    for i in 1:n+1
        MOI.addconstraint!(optimizer, MOI.SingleVariable(Xsq[i,i]), MOI.EqualTo(1.0))
    end

    objf_t = vec([MOI.ScalarAffineTerm(L[i,j], Xsq[i,j]) for i in 1:n+1, j in 1:n+1])
    MOI.set!(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))

    MOI.set!(optimizer, MOI.ObjectiveSense(), MOI.MinSense)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)

    for i in 1:n+1, j in 1:n+1
        @test 1.0001> abs(Xsq_s[i,j]) > 0.9999
    end

    mimo_eval(s,H,y,L,Xsq_s)
end