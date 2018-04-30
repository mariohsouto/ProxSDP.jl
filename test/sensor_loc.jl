
function sensor_loc(solver, seed)
    srand(seed)
    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end
    
    # Instance size
    n = 100
    m = 10 * n
    # Sensor true position
    x_true = rand((n, 1))
    # Anchor positions
    a = Dict(i => rand((n, 1)) for i in 1:m)
    d = Dict(i => norm(x_true - a[i]) for i in 1:m)
    A = Dict()
    for i in 1:m
        A[i] = [hcat(eye(n), -a[i]); hcat(-a[i]', 0.0)]
    end
    
    # Build SDP problem
    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n+1, 1:n+1], PSD)
    else
        @variable(model, X[1:n+1, 1:n+1], SDP)
    end
    # Distance constraints
    @constraint(model, ctr[i in 1:m], sum(A[i] .* X) == d[i]^2 - norm(a[i])^2)
    # Box constraints
    # @constraint(model, ub_1[i in 1:n], X[i, n + 1] <= 1.0)
    # @constraint(model, lb_1[i in 1:n], X[i, n + 1] >= 0.0)
    # @constraint(model, ub_2[i in 1:n], X[n + 1, i] <= 1.0)
    # @constraint(model, lb_2[i in 1:n], X[n + 1, i] >= 0.0)
    @constraint(model, ub_2[i in 1:n, j in 1:n], X[i, j] <= 1.0)
    @constraint(model, lb_2[i in 1:n, j in 1:n], X[i, j] >= 0.0)
    @constraint(model, X[n + 1, n + 1] == 1.0)
    # Feasibility objective function
    L = eye(n+1)
    @objective(model, Min, sum(0.0 * X[i, j] for i in 1:n+1, j in 1:n+1))

    # Solve
    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(model, solver)
    end
    tic()
    teste = JuMP.solve(model)
    toc()

    # Decode signal and show results
    if Base.libblas_name == "libmkl_rt"
        XX = getvalue2.(X)
    else
        XX = getvalue.(X)
    end
    @show norm(x_true - XX[1:n, end])
    @show rank = length([eig for eig in eigfact(XX)[:values] if eig > 1e-7])
    # @show eigfact(XX)[:values]
    return nothing
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])