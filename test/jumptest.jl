
function mimo(solver)
    srand(0)
    if VERSION > v"0.6.0"
        m = Model()
    else
        m = Model(solver=solver) 
    end
    
    # Instance size
    n = 200
    # Channel
    H = randn((n, n))
    # Gaussian noise
    v = randn((n, 1))
    # True signal
    s = rand([-1, 1], n)
    # Received signal
    sigma = 0.01
    y = H * s + sigma * v
    L = [hcat(H' * H, -H' * y); hcat(-y' * H, y' * y)]

    if VERSION > v"0.6.0"
        @variable(m, X[1:n+1, 1:n+1], PSD)
    else
        @variable(m, X[1:n+1, 1:n+1], SDP)
    end
    @objective(m, Min, sum(L[i, j] * X[i, j] for i in 1:n+1, j in 1:n+1))
    @constraint(m, ctr[i in 1:n+1], X[i, i] == 1.0)
    @constraint(m, bla, X[1, 1] <= 1.0)

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

