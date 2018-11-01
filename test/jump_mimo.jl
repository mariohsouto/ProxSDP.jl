function jump_mimo(solver, seed, n, verbose = false)

    # n = 3
    m = 10n
    s, H, y, L = mimo_data(seed, m, n)

    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end

    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n+1, 1:n+1], PSD)
    else
        @variable(model, X[1:n+1, 1:n+1], SDP)
    end
    @objective(model, Min, sum(L[i, j] * X[i, j] for i in 1:n+1, j in 1:n+1))
    @constraint(model, ctr[i in 1:n+1], X[i, i] == 1.0)

    # for i in 1:n+1
    #     for j in 1:n+1
    #         setlowerbound(X[i, j], -1.0)
    #     end
    # end

    # @constraint(model, lb[i in 1:n+1, j in i:n+1], X[i, j] >= - 1.0)
    # @constraint(model, lb[i in 1:n+1, j in 1:n+1], X[i, j] >= - 1.0)
    # @constraint(model, ub[i in 1:n+1, j in 1:n+1], X[i, j] <= 1.0)

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

    verbose && mimo_eval(s,H,y,L,XX)

    return nothing
end

# getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])