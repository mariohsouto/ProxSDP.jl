
function max_cut(solver, path)

    # Read data from file
    data = readdlm(path)

    # Instance size
    n = data[1, 1]
    # Partition weights
    W = zeros((n, n))
    for k=5:size(data)[1]
        if data[k, 1] == 0
            W[data[k, 3], data[k, 4]] = - data[k, 5]
            W[data[k, 4], data[k, 3]] = - data[k, 5]
        end
    end

    if Base.libblas_name == "libmkl_rt"
        m = Model()
    else
        m = Model(solver=solver) 
    end
    # m = Model()

    if Base.libblas_name == "libmkl_rt"
        @variable(m, X[1:n, 1:n], PSD)
    else
        @variable(m, X[1:n, 1:n], SDP)
    end
    # @variable(m, X[1:n, 1:n], PSD)
    @objective(m, Min, sum(W[i, j] * X[i, j] for i in 1:n, j in 1:n))
    @constraint(m, ctr[i in 1:n], X[i, i] == 1.0)

    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(m, solver)
    end
    # JuMP.attach(m, solver)
    @time teste = JuMP.solve(m)

    # @show JuMP.resultvalue.(X)
end