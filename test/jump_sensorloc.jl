
function jump_sensorloc(solver, seed, n; verbose = false, test = false)

    m, x_true, a, d, d_bar = sensorloc_data(seed, n)

    model = Model(with_optimizer(solver))

    # Build SDP problem
    @variable(model, X[1:n+2, 1:n+2], PSD)

    # Constraint with distances from anchors to sensors
    for j in 1:n
        for k in 1:m
            # e = zeros(n, 1)
            # e[j] = -1.0
            # v = vcat(a[k], e)
            # V = v * v'
            # @constraint(model, sum(V .* X) == d_bar[k, j]^2)
            @constraint(model, X[1,1]*a[k][1]*a[k][1] + X[2,2]*a[k][2]*a[k][2] 
                                 - 2 * X[1,2]*a[k][1]*a[k][2] 
                                 - 2 * X[1, j+2] * a[k][1]
                                 - 2 * X[2, j+2] * a[k][2]
                                 + X[j+2, j+2]
                                == d_bar[k, j]^2)
        end
    end

    # Constraint with distances from sensors to sensors
    count, count_all = 0, 0
    for i in 1:n
        for j in 1:i - 1
            count_all += 1
            if rand() > 0.9
                count += 1
                # e = zeros(n, 1)
                # e[i] = 1.0
                # e[j] = -1.0
                # v = vcat(zeros(2, 1), e)
                # V = v * v'
                # @constraint(model, sum(V .* X) == d[i, j]^2)
                @constraint(model, X[i+2,i+2] + X[j+2,j+2] - 2*X[i+2,j+2] == d[i, j]^2)
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
    @objective(model, Min, 0.0 * X[1, 1] + 0.0 * X[2, 2])
    
    teste = @time optimize!(model)

    XX = value.(X)

    verbose && sensorloc_eval(n, m, x_true, XX)

    objval = objective_value(model)
    stime = MOI.get(model, MOI.SolveTime())
    return (objval,stime)
end
