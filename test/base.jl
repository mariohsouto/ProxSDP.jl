function base_sdp(solver, seed)

    srand(seed)
    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end
    
    n = 2

    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n, 1:n], PSD)
    else
        @variable(model, X[1:n, 1:n], SDP)
    end
    @objective(model, Min, -3X[1,1]-4X[2,2])
    @constraint(model, 2X[1,1]+1X[2,2] <= 4)
    @constraint(model, 1X[1,1]+2X[2,2] <= 4)

    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(model, solver)
    end
    teste = JuMP.solve(model)

    if Base.libblas_name == "libmkl_rt"
        @show XX = getvalue2.(X)
    else
        @show XX = getvalue.(X)
    end

end
function base_sdp2(solver, seed)

    srand(seed)
    if Base.libblas_name == "libmkl_rt"
        model = Model()
    else
        model = Model(solver=solver) 
    end
    
    n = 4

    if Base.libblas_name == "libmkl_rt"
        @variable(model, X[1:n, 1:n], PSD)
    else
        @variable(model, X[1:n, 1:n], SDP)
    end
    @objective(model, Min, -3X[1,1]-4X[2,2])
    @constraint(model, 2X[1,1]+1X[2,2]+X[3,3] == 4)
    @constraint(model, 1X[1,1]+2X[2,2]+X[4,4] == 4)

    if Base.libblas_name == "libmkl_rt"
        JuMP.attach(model, solver)
    end
    teste = JuMP.solve(model)

    if Base.libblas_name == "libmkl_rt"
        @show XX = getvalue2.(X)
    else
        @show XX = getvalue.(X)
    end

end

getvalue2(var::JuMP.Variable) = (m=var.m;m.solverinstance.primal[m.solverinstance.varmap[m.variabletosolvervariable[var.instanceindex]]])
