function rand_sdp(solver, seed)
    srand(seed)
    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end
    
    n = 2  # Instance size
    m = 1  # Number of constraints
    # Objective function
    c_sqrt = rand((n, n))
    C = c_sqrt * c_sqrt'
    # C[1, 2] *= 0.5
    # C[2, 1] = C[1, 2]
    # Generate m-dimensional feasible system
    A, b = Dict(), Dict()
    X_ = randn((n, n))
    X_ = X_ * X_'
    for i in 1:m
        A[i] = rand((n, n))
        A[i] = A[i] * A[i]'
        b[i] = trace(A[i] * X_)
    end

    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n, 1:n], PSD)
    else
        @variable(model, X[1:n, 1:n], SDP)
    end
    @objective(model, Min, sum(C[i, j] * X[i, j] for i in 1:n, j in 1:n))
    @constraint(model, ctr[k in 1:m], sum(A[k][i, j] * X[i, j] for i in 1:n, j in 1:n) == b[k])
    # @constraint(model, bla, sum(C[i, j] * X[i, j] for i in 1:n, j in 1:n)<=0.1)

    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(model, solver)
    end
    teste = JuMP.solve(model)

    if Base.libblas_name == "libmkl_rt"
        XX = getvalue2.(X)
    else
        XX = getvalue.(X)
    end

    minus_rank = length([eig for eig in eigfact(XX)[:values] if eig < -1e-10])
    @show minus_rank

    rank = length([eig for eig in eigfact(XX)[:values] if eig > 1e-10])
    @show rank

    @show trace(C * XX)
    for i in 1:m
        @show trace(A[i] * XX)-b[i]
    end
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])