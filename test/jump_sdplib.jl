
function jump_sdplib(solver, path, verbose = false)
    tic()

    println("running: $(path)")

    n, m, F, c = sdplib_data(path)

    # Build model
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

    # Objective function
    @objective(model, Min, sum(F[0][idx...] * X[idx...] for idx in zip(findnz(F[0])[1:end-1]...)))

    # Linear equality constraints
    for k = 1:m
        @constraint(model, sum(F[k][idx...] * X[idx...] for idx in zip(findnz(F[k])[1:end-1]...)) == c[k])
    end

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

    verbose && sdplib_eval(F,c,n,m,XX)

    return nothing
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])