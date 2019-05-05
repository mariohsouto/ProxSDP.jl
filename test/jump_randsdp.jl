function jump_randsdp(solver, seed, n, m, verbose = false)

    A, b, C = randsdp_data(seed, m, n)

    model = Model(with_optimizer(solver))
    @variable(model, X[1:n, 1:n], PSD)
    @objective(model, Min, sum(C[i, j] * X[i, j] for j in 1:n, i in 1:n))
    @constraint(model, ctr[k in 1:m], sum(A[k][i, j] * X[i, j] for j in 1:n, i in 1:n) == b[k])
    # @constraint(model, bla, sum(C[i, j] * X[i, j] for i in 1:n, j in 1:n)<=0.1)

    teste = @time optimize!(model)

    XX = value.(X)
    
    verbose && randsdp_eval(A,b,C,n,m,XX)
    
    objval = objective_value(model)
    stime = MOI.get(model, MOI.SolveTime())
    return (objval,stime)
end
