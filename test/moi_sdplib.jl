function moi_sdplib(optimizer, path)

    println("running: $(path)")

    MOI.empty!(optimizer)
    @test MOI.isempty(optimizer)

    println("flag 1")
    n, m, F, c = sdplib_data(path)
    println("flag 2")

    nvars = ProxSDP.sympackedlen(n)

    println("flag 3")

    X = MOI.addvariables!(optimizer, nvars)
    Xsq = Matrix{MOI.VariableIndex}(n,n)
    ProxSDP.ivech!(Xsq, X)
    Xsq = full(Symmetric(Xsq,:U))

    println("flag 4")

    for k in 1:m
        I,J,V=findnz(F[k])
        ctr_k = [MOI.ScalarAffineTerm(V[ind], Xsq[I[ind],J[ind]]) for ind in eachindex(I)]
            MOI.addconstraint!(optimizer, MOI.ScalarAffineFunction(ctr_k, 0.0), MOI.EqualTo(c[k]))
    end

    println("flag 5")

    vov = MOI.VectorOfVariables(X)
    cX = MOI.addconstraint!(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n))

    println("flag 6")

    I,J,V=findnz(F[0])
    println("flag 7")
    objf_t = [MOI.ScalarAffineTerm(V[ind], Xsq[I[ind],J[ind]]) for ind in eachindex(I)]
    println("flag 8")
    MOI.set!(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))
    println("flag 9")
    MOI.set!(optimizer, MOI.ObjectiveSense(), MOI.MinSense)
    println("flag 10")
    MOI.optimize!(optimizer)
    println("flag 11")
    obj = MOI.get(optimizer, MOI.ObjectiveValue())
    println("flag 12")
    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)
    println("flag 13")
    minus_rank = length([eig for eig in eigfact(Xsq_s)[:values] if eig < -1e-4])
    @test minus_rank == 0

    @test trace(F[0] * Xsq_s) - obj < 1e-1
    for i in 1:m
        @test abs(trace(F[i] * Xsq_s)-c[i]) < 1e-1
    end

    sdplib_eval(F,c,n,m,Xsq_s)
end