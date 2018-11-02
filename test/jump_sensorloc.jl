
function jump_sensorloc(solver, seed, n, verbose)

    m, x_true, a, d, d_bar = sensorloc_data(seed, n)

    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end

    # Build SDP problem
    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n+2, 1:n+2], PSD)
    else
        @variable(model, X[1:n+2, 1:n+2], SDP)
    end

    # Constraint with distances from anchors to sensors
    for k in 1:m
        for j in 1:n
            e = zeros(n, 1)
            e[j] = -1.0
            v = vcat(a[k], e)
            V = v * v'
            @constraint(model, sum(V .* X) == d_bar[k, j]^2)
        end
    end

    # Constraint with distances from sensors to sensors
    count, count_all = 0, 0
    for i in 1:n
        for j in 1:i - 1
            count_all += 1
            if rand() > 0.9
                count += 1
                e = zeros(n, 1)
                e[i] = 1.0
                e[j] = -1.0
                v = vcat(zeros(2, 1), e)
                V = v * v'
                @constraint(model, sum(V .* X) == d[i, j]^2)
            end
        end   
    end
    if verbose
        @show count_all, count
    end

    @constraint(model, X[1, 1] == 1.0)
    @constraint(model, X[1, 2] == 0.0)
    @constraint(model, X[2, 1] == 0.0)
    @constraint(model, X[2, 2] == 1.0)

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

    verbose && sensorloc_eval(n, m, x_true, XX)

    return nothing
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])