function moi_randsdp(optimizer, seed, n, m; verbose = false, test = false, atol = 1e-2)

    MOI.empty!(optimizer)
    if test
        @test MOI.is_empty(optimizer)
    end

    A, b, C = randsdp_data(seed, m, n)

    nvars = sympackedlen(n)

    X = MOI.add_variables(optimizer, nvars)

    Xsq = Matrix{MOI.VariableIndex}(undef,n,n)
    ivech!(Xsq, X)
    Xsq = Matrix(Symmetric(Xsq,:U))

    for k in 1:m
        ctr_k = vec([MOI.ScalarAffineTerm(A[k][i,j], Xsq[i,j]) for j in 1:n, i in 1:n])
            MOI.add_constraint(optimizer, MOI.ScalarAffineFunction(ctr_k, 0.0), MOI.EqualTo(b[k]))
    end

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n))

    objf_t = vec([MOI.ScalarAffineTerm(C[i,j], Xsq[i,j]) for j in 1:n, i in 1:n])
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    objval = MOI.get(optimizer, MOI.ObjectiveValue())

    stime = -1.0
    try
        stime = MOI.get(optimizer, MOI.SolveTime())
    catch
        println("could not query time")
    end

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)

    minus_rank = length([eig for eig in eigen(Xsq_s).values if eig < -1e-4])
    if test
        @test minus_rank == 0
    end
    # rank = length([eig for eig in eigen(XX).values if eig > 1e-10])
    # @show rank
    if test
        @test tr(C * Xsq_s) - objval < atol
        for i in 1:m
            @test abs(tr(A[i] * Xsq_s)-b[i])/(1+abs(b[i])) < atol
        end
    end
    verbose && randsdp_eval(A,b,C,n,m,Xsq_s)

    rank = -1
    return (objval, stime, rank)
end