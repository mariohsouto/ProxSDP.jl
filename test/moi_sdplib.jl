function moi_sdplib(optimizer, path)

    println("running: $(path)")

    MOI.empty!(optimizer)
    @test MOI.isempty(optimizer)

    n, m, F, c = sdplib_data(path)

    nvars = ProxSDP.sympackedlen(n)

    X = MOI.addvariables!(optimizer, nvars)
    Xsq = Matrix{MOI.VariableIndex}(n,n)
    ProxSDP.ivech!(Xsq, X)
    Xsq = full(Symmetric(Xsq,:U))

    for k in 1:m
        I,J,V=findnz(F[k])
        ctr_k = [MOI.ScalarAffineTerm(V[ind], Xsq[I[ind],J[ind]]) for ind in eachindex(I)]
            MOI.addconstraint!(optimizer, MOI.ScalarAffineFunction(ctr_k, 0.0), MOI.EqualTo(c[k]))
    end

    vov = MOI.VectorOfVariables(X)
    cX = MOI.addconstraint!(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n))

    I,J,V=findnz(F[0])
    objf_t = [MOI.ScalarAffineTerm(V[ind], Xsq[I[ind],J[ind]]) for ind in eachindex(I)]
    MOI.set!(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))
    MOI.set!(optimizer, MOI.ObjectiveSense(), MOI.MinSense)
    MOI.optimize!(optimizer)
    obj = MOI.get(optimizer, MOI.ObjectiveValue())
    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)
    minus_rank = length([eig for eig in eigfact(Xsq_s)[:values] if eig < -1e-4])
    @test minus_rank == 0

    @test trace(F[0] * Xsq_s) - obj < 1e-1
    for i in 1:m
        @test abs(trace(F[i] * Xsq_s)-c[i]) < 1e-1
    end

    sdplib_eval(F,c,n,m,Xsq_s)
end