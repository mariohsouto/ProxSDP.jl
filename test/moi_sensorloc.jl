function moi_sensorloc(optimizer, seed, n)

    srand(seed)
    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)

    m, x_true, a, d, A = sensorloc_data(seed, n)

    nvars = ProxSDP.sympackedlen(n + 1)

    X = MOI.add_variables(optimizer, nvars)

    Xsq = Matrix{MOI.VariableIndex}(n+1,n+1)
    ProxSDP.ivech!(Xsq, X)
    Xsq = full(Symmetric(Xsq,:U))

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n+1))

    for k in 1:m
        ctr_aff = vec([MOI.ScalarAffineTerm(A[k][i,j], Xsq[i,j]) for i in 1:n+1, j in 1:n+1])
        MOI.add_constraint(optimizer, MOI.ScalarAffineFunction(ctr_aff, 0.0), MOI.EqualTo(d[k]^2 - norm(a[k])^2))
    end
    MOI.add_constraint(optimizer, MOI.SingleVariable(Xsq[n+1,n+1]), MOI.EqualTo(1.0))

    objf_t = [MOI.ScalarAffineTerm(0.0, Xsq[1,1])]
    if false
        objf_t = [MOI.ScalarAffineTerm(1.0, Xsq[i,i]) for i in 1:n+1]
    end
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MinSense)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)

    sensorloc_eval(n, m, x_true, Xsq_s)

    return nothing
end