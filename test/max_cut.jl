
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

    # if Base.libblas_name == "libmkl_rt"
    #     m = Model()
    # else
    #     m = Model(solver=solver) 
    # end
    m = Model()

    # if Base.libblas_name == "libmkl_rt"
    #     @variable(m, X[1:n, 1:n], PSD)
    # else
    #     @variable(m, X[1:n, 1:n], SDP)
    # end
    @variable(m, X[1:n, 1:n], PSD)
    @objective(m, Min, sum(W[i, j] * X[i, j] for i in 1:n, j in 1:n))
    @constraint(m, ctr[i in 1:n], X[i, i] == 1.0)

    # if Base.libblas_name == "libmkl_rt"
    #     JuMP.attach(m, solver)
    # end
    JuMP.attach(m, solver)
    teste = JuMP.solve(m)

    # @show JuMP.resultvalue.(X)

    # Goemans-Williamson rounding
    best_obj = inf
    for _ in 1:10
        # Draw a vector uniformly distributed in the unit sphere
        r = randn((dims.n, 1))
        r /= norm(r)
        # Define cuts S1 = {i | <x_i, v > >= 0} and S2 = {i | <x_i, v > < 0}
        S1, S2 = [], []
        for i in n
            if dot(XX[:, i], r) >= 0.0
                push!(S1, i)
            else
                push!(S2, i)
            end
        end
        # Compute value defined by the cut(S1, S2)
        obj = 0.0
        for i in 1:n
            for j in 1:n
                if (i in S1 && j in S2) || (i in S2 and j in S1)
                    obj += W[i, j]
                end
            end
        end

        # Save best incumbent
        if obj < best_obj
            best_obj = obj
        end
    end
    @show best_obj
end