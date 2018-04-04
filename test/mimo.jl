
function mimo(solver)
    srand(0)
    if Base.libblas_name == "libmkl_rt"
        m = Model()
    else
        m = Model(solver=solver) 
    end
    
    # Instance size
    n = 500
    # Channel
    H = randn((n, n))
    # Gaussian noise
    v = randn((n, 1))
    # True signal
    s = rand([-1, 1], n)
    # Received signal
    sigma = 1e-2
    y = H * s + sigma * v
    L = [hcat(H' * H, -H' * y); hcat(-y' * H, y' * y)]
    if Base.libblas_name == "libmkl_rt"
        @variable(m, X[1:n+1, 1:n+1], PSD)
    else
        @variable(m, X[1:n+1, 1:n+1], SDP)
    end
    @objective(m, Min, sum(L[i, j] * X[i, j] for i in 1:n+1, j in 1:n+1))
    @constraint(m, ctr[i in 1:n+1], X[i, i] == 1.0)

    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(m, solver)
    end

    teste = JuMP.solve(m)
    if Base.libblas_name == "libmkl_rt"
        XX = getvalue2.(X)
    else
        XX = getvalue.(X)
    end
    x_hat = sign.(XX[1:n, end])
    println(x_hat-s)
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])