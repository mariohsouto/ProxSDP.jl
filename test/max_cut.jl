
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

    if VERSION > v"0.6.0"
        m = Model()
    else
        m = Model(solver=solver) 
    end

    if VERSION > v"0.6.0"
        @variable(m, X[1:n, 1:n], PSD)
    else
        @variable(m, X[1:n, 1:n], SDP)
    end
    @objective(m, Min, sum(W[i, j] * X[i, j] for i in 1:n, j in 1:n))
    @constraint(m, ctr[i in 1:n], X[i, i] == 1.0)

    if VERSION > v"0.6.0"
        JuMP.attach(m, solver)
        # OSX
        # JuMP.attach(m, MosekInstance(
        #     MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-3, MSK_DPAR_INTPNT_CO_TOL_INFEAS=1e-3,
        #     MSK_DPAR_INTPNT_CO_TOL_MU_RED=1e-3, 
        #     MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-3, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1e-3
        # ))
    end
    teste = JuMP.solve(m)
end