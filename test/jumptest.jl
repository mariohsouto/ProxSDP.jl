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

# @testset "LP0" begin
#     m = Model()
#     @variable(m,x[1:4,1:4],PSD)
#     @constraint(m, 1x[1,1]+2x[2,2]+x[3,3]==4)
#     @constraint(m, 2x[1,1]+1x[2,2]+x[4,4]==4)
#     @objective(m, Min, 1x[1,1]+2x[2,2])
#     JuMP.attach(m, ProxSDPSolverInstance())
#     JuMP.solve(m)
# end


@testset "Linear Programming" begin
    @testset "LP1" begin

        srand(0)
        m = Model()
        n = 100
        # Channel
        H = randn((n, n))
        # Gaussian noise
        v = randn((n, 1))
        # True signal
        s = rand([-1, 1], n)
        # Received signal
        sigma = 0.01
        y = H * s + sigma * v
        L = [hcat(H' * H, -H' * y); hcat(-y' * H, y' * y)]
        # L = readcsv("/Users/mariosouto/Dropbox/proxsdp/L.csv")

        @variable(m, X[1:n+1, 1:n+1], PSD)
        @objective(m, Min, sum(L[i, j] * X[i, j] for i in 1:n+1, j in 1:n+1))
        @constraint(m, ctr[i in 1:n+1], X[i, i] == 1.0)
        @constraint(m, bla, X[1, 1] <= 10.0)

        # JuMP.attach(m, SCSInstance())
        # JuMP.attach(m, MosekInstance())
        # JuMP.attach(m, CSDP.CSDPInstance(maxiter=100000))
        JuMP.attach(m, ProxSDPSolverInstance())

        teste = JuMP.solve(m)
        println("Duals equal. : ", JuMP.resultdual.(ctr))
        println("Duals inequal. : ", JuMP.resultdual(bla))
        println("Objective value: ", JuMP.objectivevalue(m))
        println("primal Status value: ", JuMP.primalstatus(m))
        println("dual Status value: ", JuMP.dualstatus(m))
        println("term Status value: ", JuMP.terminationstatus(m))
        # println(JuMP.resultvalue.(X))
    end
end