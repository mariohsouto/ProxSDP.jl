
function jump_sdplib(solver, path; verbose = false, test = false)

    println("running: $(path)")

    n, m, F, c = sdplib_data(path)

    # Build model
    model = Model(with_optimizer(solver))
    @variable(model, X[1:n, 1:n], PSD)

    # Objective function
    @objective(model, Min, sum(F[0][idx...] * X[idx...] 
        for idx in zip(findnz(F[0])[1:end-1]...)))

    # Linear equality constraints
    for k = 1:m
        @constraint(model, sum(F[k][idx...] * X[idx...]
            for idx in zip(findnz(F[k])[1:end-1]...)) == c[k])
    end
    
    teste = @time optimize!(model)

    XX = value.(X)

    verbose && sdplib_eval(F,c,n,m,XX)

    objval = objective_value(model)
    stime = MOI.get(model, MOI.SolveTime())


    # @show tp = typeof(model.moi_backend.optimizer.model.optimizer)
    # @show fieldnames(tp)
    rank = -1
    try
        @show rank = model.moi_backend.optimizer.model.optimizer.sol.final_rank
    catch
    end
    return (objval, stime, rank)
    # return (objval, stime)
end
