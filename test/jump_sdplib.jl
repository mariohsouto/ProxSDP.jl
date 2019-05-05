
function jump_sdplib(solver, path; verbose = false, test = false)

    println("running: $(path)")

    n, m, F, c = sdplib_data(path)

    # Build model
    model = Model(with_optimizer(solver))
    @variable(model, X[1:n, 1:n], PSD)

    # Objective function
    @objective(model, Min, sum(F[0][idx...] * X[idx...] for idx in zip(findnz(F[0])[1:end-1]...)))

    # Linear equality constraints
    for k = 1:m
        @constraint(model, sum(F[k][idx...] * X[idx...] for idx in zip(findnz(F[k])[1:end-1]...)) == c[k])
    end
    
    teste = @time optimize!(model)

    XX = value.(X)

    verbose && sdplib_eval(F,c,n,m,XX)

    return nothing
end
