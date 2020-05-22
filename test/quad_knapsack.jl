using StatsBase

function quad_knapsack(solver, seed)
    rng = Random.MersenneTwister(seed)
    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end
    
    # Instance size
    n = 20
    # k-item capacity
    # k = Int(n / 10)
    k = 1
    # Frequency of nonzero weights
    delta = 0.5

    # Build weights and capacity
    a = zeros(n)
    for i in 1:n
        a[i] = rand(rng, 1:50)
    end
    a = ones(n)
    b = rand(100:sum(a)+100)

    # Profits
    C = zeros((n, n))
    for i in 1:n
        for j in 1:n
            if sample(rng, [1, 0],Weights([delta, 1.0 - delta])) == 1
                c_ = - rand(rng, 1:100)
                C[i, j] = c_
                C[j, i] = c_
            end
        end
    end

    # Decision variable
    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n+1, 1:n+1], PSD)
    else
        @variable(model, X[1:n+1, 1:n+1], SDP)
    end
    @objective(model, Min, sum(C[i, j] * X[i+1, j+1] for i in 1:n, j in 1:n))
    # Capacity constraint
    # @constraint(model, cap, sum(a[i] * X[i+1, i+1] for i in 1:n) <= b)
    # k-item constraint
    @constraint(model, k_knap, sum(X[i+1, i+1] for i in 1:n) == k)

    # @constraint(model, bla, X[1, 1] == 1)

    # @constraint(model, lb[i in 1:n+1, j in 1:n+1], X[i, j] >= 0.0)
    # @constraint(model, ub[i in 1:n+1, j in 1:n+1], X[i, j] <= 1.0)

    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(model, solver)
    end
    
    @time teste = JuMP.solve(model)
    
    if Base.libblas_name == "libmkl_rt"
        XX = getvalue2.(X)
    else
        XX = getvalue.(X)
    end
    rank = length([eig for eig in eigen(XX).values if eig > 1e-10])
    @show rank
    @show diag(XX)
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])