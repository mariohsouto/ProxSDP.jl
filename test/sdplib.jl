
function sdplib(solver, path)
    tic()
    # Read data from file
    data = readdlm(path)

    # Parse SDPLIB data
    m = data[1, 1]
    if isa(data[3, 1], Float64) || isa(data[3, 1], Int64)
        blks = data[3, :]
        for elem = 1:length(blks)
            if blks[elem] == ""
                blks = blks[1:elem-1]
                break
            end
        end
    else
        blks = parse.(Float64, split(data[3, 1][2:end - 1], ","))
    end
    cum_blks = unshift!(cumsum(blks), 0)
    if isa(data[4, 1], Float64) || isa(data[4, 1], Int64) 
        c = data[4, :]
    else
        c = [parse(Float64,string) for string in split(data[4, 1][2:end - 1], ",")]
    end
    # n = abs(cum_blks[end])
    n = length(c)
    n = 335
    F = Dict(i => spzeros(n, n) for i = 0:m)
    for k=5:size(data)[1]
        idx = cum_blks[data[k, 2]]
        if data[k, 2] == 1
            idx = 0
        else
            idx = 161
        end

        i, j = data[k, 3] + idx, data[k, 4] + idx
        if data[k, 1] == 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            F[0][i, j] = - data[k, 5]
            F[0][j, i] = - data[k, 5]
        else
            F[data[k, 1]][i, j] = data[k, 5]
            F[data[k, 1]][j, i] = data[k, 5]
        end
    end
    println("build F")

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
    println("obj")

    # Linear equality constraints
    for k = 1:m
        @constraint(model, sum(F[k][idx...] * X[idx...] for idx in zip(findnz(F[k])[1:end-1]...)) == c[k])
    end
    println("const")

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
    rank = length([eig for eig in eigfact(XX)[:values] if eig > 1e-4])
    @show rank
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])