function jump_randsdp(solver, seed, n, m, verbose = false)

    A, b, C = randsdp_data(seed, m, n)

    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end

    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n, 1:n], PSD)
    else
        @variable(model, X[1:n, 1:n], SDP)
    end
    @objective(model, Min, sum(C[i, j] * X[i, j] for i in 1:n, j in 1:n))
    @constraint(model, ctr[k in 1:m], sum(A[k][i, j] * X[i, j] for i in 1:n, j in 1:n) == b[k])
    # @constraint(model, bla, sum(C[i, j] * X[i, j] for i in 1:n, j in 1:n)<=0.1)

    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(model, solver)
    end
    teste = JuMP.solve(model)

    if Base.libblas_name == "libmkl_rt"
        XX = getvalue2.(X)
    else
        XX = getvalue.(X)
    end

    verbose && randsdp_eval(A,b,C,n,m,XX)

    return nothing
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])