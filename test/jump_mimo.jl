function jump_mimo(solver, seed, n; verbose = false, test = false)

    # n = 3
    m = 10n
    s, H, y, L = mimo_data(seed, m, n)

    nvars = ProxSDP.sympackedlen(n + 1)

    model = Model(with_optimizer(solver))
    @variable(model, X[1:n+1, 1:n+1], PSD)
    for j in 1:(n+1), i in j:(n+1)
        @constraint(model, X[i, j] <=  1.0)
        @constraint(model, X[i, j] >= -1.0)
    end
    @objective(model, Min, sum(L[i, j] * X[i, j] for j in 1:n+1, i in 1:n+1))
    @constraint(model, ctr[i in 1:n+1], X[i, i] == 1.0)

    teste = @time optimize!(model)

    XX = value.(X)

    if test
        for j in 1:n+1, i in 1:n+1
            @test 1.01 > abs(XX[i,j]) > 0.99
        end
    end

    verbose && mimo_eval(s,H,y,L,XX)

    objval = objective_value(model)
    stime = MOI.get(model, MOI.SolveTimeSec())

    rank = -1
    try
        @show rank = model.moi_backend.optimizer.model.optimizer.sol.final_rank
    catch
    end
    status = 0
    if JuMP.termination_status(model) == MOI.OPTIMAL
        status = 1
    end

    # SDP constraints
    max_spd_violation = minimum(eigen(XX).values)

    # test violations of linear constraints
    max_lin_viol = 0.0
    for j in 1:(n+1), i in j:(n+1)
        val = abs(XX[i,j]) - 1.0
        if val > 0.0
            if val > max_lin_viol
                max_lin_viol = val
            end
        end
    end
    for j in 1:(n+1)
        val = abs(XX[j,j] - 1.0)
        if val > 0.0
            if val > max_lin_viol
                max_lin_viol = val
            end
        end
    end

    return (objval, stime, rank, status, max_lin_viol, max_spd_violation)
end
