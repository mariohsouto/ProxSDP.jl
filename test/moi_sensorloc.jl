function moi_sensorloc(optimizer, seed, n)

    srand(seed)
    MOI.empty!(optimizer)
    @test MOI.isempty(optimizer)

    # Instance size
    m = 10 * n
    # Sensor true position
    x_true = rand((n, 1))
    # Anchor positions
    a = Dict(i => rand((n, 1)) for i in 1:m)
    d = Dict(i => norm(x_true - a[i]) for i in 1:m)
    A = Dict()
    for i in 1:m
        A[i] = [hcat(eye(n), -a[i]); hcat(-a[i]', 0.0)]
    end

    nvars = ProxSDP.sympackedlen(n + 1)

    X = MOI.addvariables!(optimizer, nvars)

    Xsq = Matrix{MOI.VariableIndex}(n+1,n+1)
    ProxSDP.ivech!(Xsq, X)
    Xsq = full(Symmetric(Xsq,:U))

    vov = MOI.VectorOfVariables(X)
    cX = MOI.addconstraint!(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n+1))

    for k in 1:m
        ctr_aff = vec([MOI.ScalarAffineTerm(A[k][i,j], Xsq[i,j]) for i in 1:n+1, j in 1:n+1])
        MOI.addconstraint!(optimizer, MOI.ScalarAffineFunction(ctr_aff, 0.0), MOI.EqualTo(d[k]^2 - norm(a[k])^2))
    end
    MOI.addconstraint!(optimizer, MOI.SingleVariable(Xsq[n+1,n+1]), MOI.EqualTo(1.0))

    objf_t = [MOI.ScalarAffineTerm(0.0, Xsq[1,1])]
    if false
        objf_t = [MOI.ScalarAffineTerm(1.0, Xsq[i,i]) for i in 1:n+1]
    end
    MOI.set!(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))

    MOI.set!(optimizer, MOI.ObjectiveSense(), MOI.MinSense)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)

    @show norm(x_true - Xsq_s[1:n, end])
    @show rank = length([eig for eig in eigfact(Xsq_s)[:values] if eig > 1e-7])

    return nothing
end