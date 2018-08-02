function jump_sensorloc(solver, seed, n)

    m, x_true, a, d, A = sensorloc_data(seed, n)

    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end

    # Build SDP problem
    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n+1, 1:n+1], PSD)
    else
        @variable(model, X[1:n+1, 1:n+1], SDP)
    end
    # Distance constraints
    @constraint(model, ctr[i in 1:m], sum(A[i] .* X) == d[i]^2 - norm(a[i])^2)
    #
    @constraint(model, X[n + 1, n + 1] == 1.0)

    # Feasibility objective function
    # L = eye(n+1)
    # @objective(model, Min, sum(L[i, j] * X[i, j] for i in 1:n+1, j in 1:n+1))
    @objective(model, Min, sum(0.0 * X[i, j] for i in 1:n+1, j in 1:n+1))

    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(model, solver)
    end

    tic()
    teste = JuMP.solve(model)
    toc()
    if Base.libblas_name == "libmkl_rt"
        XX = getvalue2.(X)
    else
        XX = getvalue.(X)
    end

    sensorloc_eval(n, m, x_true, XX)

    return nothing
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])