#push!(Base.LOAD_PATH,joinpath(dirname(@__FILE__),"..",".."))
# using ProxSDP
using JuMP
# using Mosek
using Base.Test
import Base.isempty
# using MathOptInterfaceSCS
# using MathOptInterfaceMosek

using MathOptInterface
const MOI = MathOptInterface
using MathOptInterfaceUtilities
const MOIU = MathOptInterfaceUtilities
# using CSDP
# using SCS
# using Mosek

@testset "Max-Cut" begin

    # Read data from file
    # data = readdlm("data/maxG55.dat-s")
    data = readdlm("data/mcp250-1.dat-s")

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

    # Build model
    m = Model() 
    # m = Model(solver=MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10000)) 

    # m = Model(solver=CSDPSolver()) 
    # m = Model(solver=MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10000)) 
    # m = Model(solver=SCSSolver())   
    
    @variable(m, X[1:n, 1:n], PSD)
    # @variable(m, X[1:n, 1:n], SDP)
    @objective(m, Min, sum(W[i, j] * X[i, j] for i in 1:n, j in 1:n))
    @constraint(m, ctr[i in 1:n], X[i, i] == 1.0)
    # @constraint(m, bla, X[1, 1] <= 10.0)

    # JuMP.attach(m, SCSInstance())
    # JuMP.attach(m, MosekInstance(
    #     MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-5, MSK_DPAR_INTPNT_CO_TOL_INFEAS=1e-5,
    #     MSK_DPAR_INTPNT_CO_TOL_MU_RED=1e-5, 
    #     MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-5, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1e-5
    # ))
    # JuMP.attach(m, CSDP.CSDPInstance(maxiter=100000))
    JuMP.attach(m, ProxSDPSolverInstance())
    tic()
    teste = JuMP.solve(m)
    println(toc())
    # println("Duals equal. : ", JuMP.resultdual.(ctr))
    # println("Objective value: ", JuMP.objectivevalue(m))
    # println("primal Status value: ", JuMP.primalstatus(m))
    # println("dual Status value: ", JuMP.dualstatus(m))
    # println("term Status value: ", JuMP.terminationstatus(m))

    # bla = Symmetric(JuMP.resultvalue.(X))
    # fact = eigfact!(bla, 0.0, Inf)
    # println(length(fact[:values][fact[:values] .> 1e-5]))
    # println(fact[:values])
end