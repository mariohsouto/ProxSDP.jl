
function mimo(solver, seed, SNR)
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
    sigma = 1.0 / SNR

    # Build SDP problem
    y = H * s + sigma * v
    L = [hcat(H' * H, -H' * y); hcat(-y' * H, y' * y)]
    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n+1, 1:n+1], PSD)
    else
        @variable(model, X[1:n+1, 1:n+1], SDP)
    end
    @objective(model, Min, sum(L[i, j] * X[i, j] for i in 1:n+1, j in 1:n+1))
    @constraint(model, ctr[i in 1:n+1], X[i, i] == 1.0)

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
    x_hat = sign.(XX[1:n, end])
    rank = length([eig for eig in eigfact(XX)[:values] if eig > 1e-7])
    # @show eigfact(XX)[:values]
    @show SNR
    @show s
    @show decode_error = sum(abs.(x_hat - s))
    @show rank
    @show norm(y - H * x_hat)
    @show norm(y - H * s)
    @show trace(L * XX) 
    return rank, decode_error
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])