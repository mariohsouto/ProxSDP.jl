
function mimo(solver, seed)
    srand(seed)
    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end
    
    # Instance size
    n = 100
    m = 10 * n
    # Channel
    H = randn((m, n))
    # Gaussian noise
    v = randn((m, 1))
    # True signal
    s = rand([-1, 1], n)
    # Received signal
    sigma = 10.0
    y = H * s + sigma * v
    L = [hcat(H' * H, -H' * y); hcat(-y' * H, y' * y)]
    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n+1, 1:n+1], PSD)
    else
        @variable(model, X[1:n+1, 1:n+1], SDP)
    end
    @objective(model, Min, sum(L[i, j] * X[i, j] for i in 1:n+1, j in 1:n+1))
    @constraint(model, ctr[i in 1:n+1], X[i, i] == 1.0)

    @constraint(model, lb[i in 1:n+1, j in 1:n+1], X[i, j] >= - 1.0)
    @constraint(model, ub[i in 1:n+1, j in 1:n+1], X[i, j] <= 1.0)

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
    @show XX[1:n, end]
    x_hat = sign.(XX[1:n, end])
    println(x_hat-s)
    rank = length([eig for eig in eigfact(XX)[:values] if eig > 1e-4])
    @show rank
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])