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

# using SemidefiniteOptInterface
# using CSDP

@testset "Max-Cut" begin

    # Read data from file
    # data = readdlm("data/maxG11.dat-s")
    data = readdlm("data/mcp500-1.dat-s")
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
    # m = Model(solver=CSDPSolver())   
    
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

    teste = JuMP.solve(m)
    println("Duals equal. : ", JuMP.resultdual.(ctr))
    println("Duals inequal. : ", JuMP.resultdual(bla))
    println("Objective value: ", JuMP.objectivevalue(m))
    println("primal Status value: ", JuMP.primalstatus(m))
    println("dual Status value: ", JuMP.dualstatus(m))
    println("term Status value: ", JuMP.terminationstatus(m))
    println(JuMP.resultvalue.(X))
end