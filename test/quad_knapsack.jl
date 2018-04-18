
function quad_knapsack(solver)
    srand(0)
    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end
    
    # Instance size
    n = 50
    k = 10
    delta = 0.5

    # Weights and capacity
    a = zeros(n)
    for i in 1:n
        a[i] = rand(1:50)
    end
    b = rand(50:sum(a))

    # Profits
    C = zeros((n, n))
    for i in 1:n
        for j in 1:n
            if rand(0:1) == 1
                c_ = - rand(1:100)
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
    @constraint(model, cap, sum(a[i] * X[i+1, i+1] for i in 1:n) <= b)
    # k-item constraint
    @constraint(model, k_knap, sum(X[i+1, i+1] for i in 1:n) == 100)

    @constraint(model, bla, X[1, 1] == 1)

    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(model, solver)
    end

    teste = JuMP.solve(model)
    if Base.libblas_name == "libmkl_rt"
        XX = getvalue2.(X)
    else
        XX = getvalue.(X)
    end
    rank = length([eig for eig in eigfact(XX)[:values] if eig > 1e-8])
    @show rank
    @show diag(XX)
end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])