function moi_randsdp(optimizer, seed, n, m; verbose = false, test = false, atol = 1e-2)

    MOI.empty!(optimizer)
    if test
        @test MOI.is_empty(optimizer)
    end

    A, b, C = randsdp_data(seed, m, n)

    nvars = ProxSDP.sympackedlen(n)

    X = MOI.add_variables(optimizer, nvars)

    Xsq = Matrix{MOI.VariableIndex}(n,n)
    ProxSDP.ivech!(Xsq, X)
    Xsq = full(Symmetric(Xsq,:U))

    for k in 1:m
        ctr_k = vec([MOI.ScalarAffineTerm(A[k][i,j], Xsq[i,j]) for j in 1:n, i in 1:n])
            MOI.add_constraint(optimizer, MOI.ScalarAffineFunction(ctr_k, 0.0), MOI.EqualTo(b[k]))
    end

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n))

    objf_t = vec([MOI.ScalarAffineTerm(C[i,j], Xsq[i,j]) for j in 1:n, i in 1:n])
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MinSense)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)

    minus_rank = length([eig for eig in eigfact(Xsq_s)[:values] if eig < -1e-4])
    if test
        @test minus_rank == 0
    end
    # rank = length([eig for eig in eigfact(XX)[:values] if eig > 1e-10])
    # @show rank
    if test
        @test trace(C * Xsq_s) - obj < atol
        for i in 1:m
            @test abs(trace(A[i] * Xsq_s)-b[i])/(1+abs(b[i])) < atol
        end
    end
    verbose && randsdp_eval(A,b,C,n,m,Xsq_s)

    return ProxSDP.get_solution(optimizer)
end