function moi_sdplib(optimizer, path; verbose = false, test = false, scalar = false)

    if verbose
        println("running: $(path)")
    end
    MOI.empty!(optimizer)
    if test
        @test MOI.is_empty(optimizer)
    end

    n, m, F, c = sdplib_data(path)

    nvars = ProxSDP.sympackedlen(n)

    X = MOI.add_variables(optimizer, nvars)
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n))

    Xsq = Matrix{MOI.VariableIndex}(undef, n,n)
    ProxSDP.ivech!(Xsq, X)
    Xsq = Matrix(LinearAlgebra.Symmetric(Xsq,:U))

    # Objective function
    objf_t = [MOI.ScalarAffineTerm(F[0][idx...], Xsq[idx...])
        for idx in zip(SparseArrays.findnz(F[0])[1:end-1]...)]
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # Linear equality constraints
    for k in 1:m
        ctr_k = [MOI.ScalarAffineTerm(F[k][idx...], Xsq[idx...]) 
            for idx in zip(SparseArrays.findnz(F[k])[1:end-1]...)]
        if scalar
            MOI.add_constraint(optimizer, MOI.ScalarAffineFunction(ctr_k, 0.0), MOI.EqualTo(c[k]))
        else
            MOI.add_constraint(optimizer,
                MOI.VectorAffineFunction(MOI.VectorAffineTerm.([1], ctr_k), [-c[k]]), MOI.Zeros(1))
        end
    end

    MOI.optimize!(optimizer)

    objval = MOI.get(optimizer, MOI.ObjectiveValue())

    stime = -1.0
    try
        stime = MOI.get(optimizer, MOI.SolveTimeSec())
    catch
        println("could not query time")
    end

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)
    minus_rank = length([eig for eig in LinearAlgebra.eigen(Xsq_s).values if eig < -1e-4])
    if test
        @test minus_rank == 0
    end
    # @test tr(F[0] * Xsq_s) - obj < 1e-1
    # for i in 1:m
    #     @test abs(tr(F[i] * Xsq_s)-c[i]) < 1e-1
    # end

    verbose && sdplib_eval(F,c,n,m,Xsq_s)

    rank = -1
    status = 0
    if MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        status = 1
    end
    return (objval, stime, rank, status)
end